# Import modules
import argparse
import os
import torch
import numpy as np

import normflow as nf
import boltzgen as bg

from time import time
from fab.utils.training import load_config
from fab.target_distributions.aldp import AldpBoltzmann
from fab import FABModel
from fab.wrappers.normflow import WrappedNormFlowModel
from fab.sampling_methods.transition_operators import HamiltonianMonteCarlo, Metropolis
from fab.utils.aldp import evaluate_aldp
from fab.utils.aldp import filter_chirality
from fab.utils.numerical import effective_sample_size
from fab.utils.replay_buffer import ReplayBuffer
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer



# Parse input arguments
parser = argparse.ArgumentParser(description='Train Boltzmann Generator with varying '
                                             'base distribution')

parser.add_argument('--config', type=str, default='../config/bm.yaml',
                    help='Path config file specifying model '
                         'architecture and training procedure')
parser.add_argument("--resume", action="store_true",
                    help='Flag whether to resume training')
parser.add_argument("--tlimit", type=float, default=None,
                    help='Number of hours after which to stop training')
parser.add_argument('--mode', type=str, default='gpu',
                    help='Compute mode, can be cpu, or gpu')
parser.add_argument('--precision', type=str, default='double',
                    help='Precision to be used for computation, '
                         'can be float, double, or mixed')

args = parser.parse_args()

# Load config
config = load_config(args.config)

# Precision
if args.precision == 'double':
    torch.set_default_dtype(torch.float64)

# Set seed
seed = config['training']['seed']
torch.manual_seed(seed)

# GPU usage
use_gpu = not args.mode == 'cpu' and torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

# Load data
path = config['data']['test']
test_data = torch.load(path)
if args.precision == 'double':
    test_data = test_data.double()
else:
    test_data = test_data.float()
test_data = test_data.to(device)


# Set up model

# Target distribution
transform_mode = 'mixed' if not 'transform' in config['system'] \
    else config['system']['transform']
shift_dih = False if not 'shift_dih' in config['system'] \
    else config['system']['shift_dih']
env = 'vacuum' if not 'env' in config['system'] \
    else config['system']['env']
ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
target = AldpBoltzmann(data_path=config['data']['transform'],
                       temperature=config['system']['temperature'],
                       energy_cut=config['system']['energy_cut'],
                       energy_max=config['system']['energy_max'],
                       n_threads=config['system']['n_threads'],
                       transform=transform_mode,
                       ind_circ_dih=ind_circ_dih,
                       shift_dih=shift_dih,
                       env=env)
target = target.to(device)

# Flow parameters
flow_type = config['flow']['type']
ndim = 60

ncarts = target.coordinate_transform.transform.len_cart_inds
permute_inv = target.coordinate_transform.transform.permute_inv.cpu().numpy()
dih_ind_ = target.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
std_dih = target.coordinate_transform.transform.ic_transform.std_dih.cpu()

ind = np.arange(ndim)
ind = np.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=np.int), ind[3 * ncarts - 6:]])
ind = ind[permute_inv]
dih_ind = ind[dih_ind_]

ind_circ = dih_ind[ind_circ_dih]
bound_circ = np.pi / std_dih[ind_circ_dih]

tail_bound = 5. * torch.ones(ndim)
tail_bound[ind_circ] = bound_circ

circ_shift = None if not 'circ_shift' in config['flow'] \
    else config['flow']['circ_shift']

# Base distribution
if config['flow']['base']['type'] == 'gauss':
    base = nf.distributions.DiagGaussian(ndim,
                                         trainable=config['flow']['base']['learn_mean_var'])
elif config['flow']['base']['type'] == 'gauss-uni':
    base_scale = torch.ones(ndim)
    base_scale[ind_circ] = bound_circ * 2
    base = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
    base.shape = (ndim,)
else:
    raise NotImplementedError('The base distribution ' + config['flow']['base']['type']
                              + ' is not implemented')

# Flow layers
layers = []
n_layers = config['flow']['blocks']

for i in range(n_layers):
    if flow_type == 'rnvp':
        # Coupling layer
        hl = config['flow']['hidden_layers'] * [config['flow']['hidden_units']]
        scale_map = config['flow']['scale_map']
        scale = scale_map is not None
        if scale_map == 'tanh':
            output_fn = 'tanh'
            scale_map = 'exp'
        else:
            output_fn = None
        param_map = nf.nets.MLP([(ndim + 1) // 2] + hl + [(ndim // 2) * (2 if scale else 1)],
                                init_zeros=config['flow']['init_zeros'], output_fn=output_fn)
        layers.append(nf.flows.AffineCouplingBlock(param_map, scale=scale,
                                                   scale_map=scale_map))
    elif flow_type == 'circular-ar-nsf':
        bl = config['flow']['blocks_per_layer']
        hu = config['flow']['hidden_units']
        nb = config['flow']['num_bins']
        ii = config['flow']['init_identity']
        dropout = config['flow']['dropout']
        layers.append(nf.flows.CircularAutoregressiveRationalQuadraticSpline(ndim,
            bl, hu, ind_circ, tail_bound=tail_bound, num_bins=nb, permute_mask=True,
            init_identity=ii, dropout_probability=dropout))
    elif flow_type == 'circular-coup-nsf':
        bl = config['flow']['blocks_per_layer']
        hu = config['flow']['hidden_units']
        nb = config['flow']['num_bins']
        ii = config['flow']['init_identity']
        dropout = config['flow']['dropout']
        if i % 2 == 0:
            mask = nf.utils.masks.create_random_binary_mask(ndim, seed=seed + i)
        else:
            mask = 1 - mask
        layers.append(nf.flows.CircularCoupledRationalQuadraticSpline(ndim,
            bl, hu, ind_circ, tail_bound=tail_bound, num_bins=nb, init_identity=ii,
            dropout_probability=dropout, mask=mask))
    else:
        raise NotImplementedError('The flow type ' + flow_type + ' is not implemented.')

    if config['flow']['mixing'] == 'affine':
        layers.append(nf.flows.InvertibleAffine(ndim, use_lu=True))
    elif config['flow']['mixing'] == 'permute':
        layers.append(nf.flows.Permute(ndim))

    if config['flow']['actnorm']:
        layers.append(nf.flows.ActNorm(ndim))

    if i % 2 == 1 and i != n_layers - 1:
        if circ_shift == 'constant':
            layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ,
                                                 shift=bound_circ))
        elif circ_shift == 'random':
            gen = torch.Generator().manual_seed(seed + i)
            shift_scale = torch.rand([], generator=gen) + 0.5
            layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ,
                                                 shift=shift_scale * bound_circ))

    # SNF
    if 'snf' in config['flow']:
        if (i + 1) % config['flow']['snf']['every_n'] == 0:
            prop_scale = config['flow']['snf']['proposal_std'] * np.ones(ndim)
            steps = config['flow']['snf']['steps']
            proposal = nf.distributions.DiagGaussianProposal((ndim,), prop_scale)
            lam = (i + 1) / n_layers
            dist = nf.distributions.LinearInterpolation(target, base, lam)
            layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

# Map input to periodic interval
layers.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))

# NormFlow model
flow = nf.NormalizingFlow(base, layers)
wrapped_flow = WrappedNormFlowModel(flow).to(device)

# Transition operator
transition_type = config['fab']['transition_type']
if transition_type == 'hmc':
    # very lightweight HMC.
    transition_operator = HamiltonianMonteCarlo(
        n_ais_intermediate_distributions=config['fab']['n_int_dist'],
        dim=ndim, L=config['fab']['n_inner'],
        epsilon=config['fab']['epsilon'] / 2,
        common_epsilon_init_weight=config['fab']['epsilon'] / 2)
    if not config['fab']['adjust_step_size']:
        transition_operator.set_eval_mode(True)
elif transition_type == 'metropolis':
    transition_operator = Metropolis(n_transitions=config['fab']['n_int_dist'],
                                     n_updates=config['fab']['n_inner'],
                                     max_step_size=config['fab']['max_step_size'],
                                     min_step_size=config['fab']['min_step_size'],
                                     adjust_step_size=config['fab']['adjust_step_size'])
else:
    raise NotImplementedError('The transition operator ' + config['fab']['transition_type']
                              + ' is not implemented')
transition_operator = transition_operator.to(device)

# FAB model
loss_type = 'alpha_2_div' if 'loss_type' not in config['fab'] \
    else config['fab']['loss_type']
model = FABModel(flow=wrapped_flow,
                 target_distribution=target,
                 n_intermediate_distributions=config['fab']['n_int_dist'],
                 transition_operator=transition_operator,
                 loss_type=loss_type)

# Prepare output directories
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
plot_dir_flow = os.path.join(plot_dir, 'flow')
plot_dir_ais = os.path.join(plot_dir, 'ais')
log_dir = os.path.join(root, 'log')
log_dir_flow = os.path.join(log_dir, 'flow')
log_dir_ais = os.path.join(log_dir, 'ais')
# Create dirs if not existent
for dir in [cp_dir, plot_dir, log_dir, plot_dir_flow,
            plot_dir_ais, log_dir_flow, log_dir_ais]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

# Initialize optimizer and its parameters
lr = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
optimizer_name = 'adam' if not 'optimizer' in config['training'] \
    else config['training']['optimizer']
optimizer_param = model.parameters()
if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(optimizer_param, lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(optimizer_param, lr=lr, weight_decay=weight_decay)
else:
    raise NotImplementedError('The optimizer ' + optimizer_name + ' is not implemented.')

if 'lr_scheduler' in config['training']:
    if config['training']['lr_scheduler']['type'] == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
            gamma=config['training']['lr_scheduler']['rate_decay'])
        lr_step = config['training']['lr_scheduler']['decay_iter']
    elif config['training']['lr_scheduler']['type'] == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
            T_max=config['training']['max_iter'])
        lr_step = 1
    elif config['training']['lr_scheduler']['type'] == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
            T_0=config['training']['lr_scheduler']['restart_iter'])
        lr_step = 1
else:
    lr_scheduler = None

lr_warmup = 'warmup_iter' in config['training'] \
            and config['training']['warmup_iter'] is not None
if lr_warmup:
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda s: min(1., s / config['training']['warmup_iter']))


# Train model
max_iter = config['training']['max_iter']
log_iter = config['training']['log_iter']
checkpoint_iter = config['training']['checkpoint_iter']

batch_size = config['training']['batch_size']
loss_hist = np.zeros((0, 2))
ess_hist = np.zeros((0, 3))
eval_samples = config['training']['eval_samples']
eval_samples_flow = len(test_data)
filter_chirality_eval = 'eval' in config['training']['filter_chirality']
filter_chirality_train = 'train' in config['training']['filter_chirality']

max_grad_norm = None if not 'max_grad_norm' in config['training'] \
    else config['training']['max_grad_norm']
grad_clipping = max_grad_norm is not None
if grad_clipping:
    grad_norm_hist = np.zeros((0, 2))

# Load train data if needed
lam_fkld = None if not 'lam_fkld' in config['fab'] else config['fab']['lam_fkld']
if loss_type == 'flow_forward_kl' or lam_fkld is not None:
    path = config['data']['train']
    train_data = torch.load(path)
    if args.precision == 'double':
        train_data = train_data.double()
    else:
        train_data = train_data.float()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=4)
    train_iter = iter(train_loader)

# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = bg.utils.get_latest_checkpoint(cp_dir, 'model')
    if latest_cp is not None:
        # Load model
        model.load(latest_cp)
        start_iter = int(latest_cp[-10:-3])
        # Load optimizer
        optimizer_path = os.path.join(cp_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        # Load scheduler
        lr_scheduler_path = os.path.join(cp_dir, 'lr_scheduler.pt')
        if lr_scheduler is not None and os.path.exists(lr_scheduler_path):
            lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))
        warmup_scheduler_path = os.path.join(cp_dir, 'warmup_scheduler.pt')
        if os.path.exists(warmup_scheduler_path):
            warmup_scheduler.load_state_dict(torch.load(warmup_scheduler_path))
        # Load logs
        log_labels = ['loss', 'ess']
        log_hists = [loss_hist, ess_hist]
        if grad_clipping:
            log_labels.append('grad_norm')
            log_hists.append(grad_norm_hist)
        for log_label, log_hist in zip(log_labels, log_hists):
            log_path = os.path.join(log_dir, log_label + '.csv')
            if os.path.exists(log_path):
                log_hist_ = np.loadtxt(log_path, delimiter=',', skiprows=1)
                if log_hist_.ndim == 1:
                    log_hist_ = log_hist_[None, :]
                log_hist.resize(*log_hist_.shape, refcheck=False)
                log_hist[:, :] = log_hist_
                log_hist.resize(np.sum(log_hist_[:, 0] <= start_iter), log_hist_.shape[1],
                                refcheck=False)

# Setup replay buffer
if 'replay_buffer' in config['training']:
    use_rb = True
    rb_config = config['training']['replay_buffer']
    if rb_config['type'] == 'uniform':
        def initial_sampler():
            x, log_w = model.annealed_importance_sampler.sample_and_log_weights(
                batch_size, logging=False)
            return x, log_w
        buffer = ReplayBuffer(dim=ndim, max_length=rb_config['max_length'] * batch_size,
                              min_sample_length=rb_config['min_length'] * batch_size,
                              initial_sampler=initial_sampler, device=str(device))
    elif rb_config['type'] == 'prioritised':
        buffer_path = os.path.join(cp_dir, 'buffer.pt')
        if os.path.exists(buffer_path):
            initial_sampler = lambda: (torch.zeros(batch_size, ndim),
                                       torch.zeros(batch_size), torch.ones(batch_size))
        else:
            def initial_sampler():
                x, log_w = model.annealed_importance_sampler.sample_and_log_weights(
                    batch_size, logging=False)
                log_q_x = model.flow.log_prob(x)
                return x, log_w, log_q_x
        buffer = PrioritisedReplayBuffer(dim=ndim, max_length=rb_config['max_length'] * batch_size,
                                         min_sample_length=rb_config['min_length'] * batch_size,
                                         initial_sampler=initial_sampler, device=str(device))
        if os.path.exists(buffer_path):
            buffer.load(buffer_path)
        # Set target distribution
        def ais_target_log_prob(x):
            return 2 * model.target_distribution.log_prob(x) - model.flow.log_prob(x)
        model.annealed_importance_sampler.target_log_prob = ais_target_log_prob
else:
    use_rb = False
    if filter_chirality_train:
        if loss_type == 'alpha_2_div':
            def modified_loss(bs):
                if isinstance(model.annealed_importance_sampler.transition_operator, HamiltonianMonteCarlo):
                    x_ais, log_w_ais = model.annealed_importance_sampler.sample_and_log_weights(bs)
                else:
                    with torch.no_grad():
                        x_ais, log_w_ais = model.annealed_importance_sampler.sample_and_log_weights(bs)
                x_ais = x_ais.detach()
                log_w_ais = log_w_ais.detach()
                ind_L = filter_chirality(x_ais)
                if torch.mean(1. * ind_L) > 0.1:
                    x_ais = x_ais[ind_L, :]
                    log_w_ais = log_w_ais[ind_L]
                loss = model.fab_alpha_div_loss_inner(x_ais, log_w_ais)
                return loss
            model.loss = modified_loss
        elif loss_type == 'flow_reverse_kl':
            def modified_loss(bs):
                x, log_q = model.flow.sample_and_log_prob((bs,))
                ind_L = filter_chirality(x)
                if torch.mean(1. * ind_L) > 0.1:
                    x = x[ind_L, :]
                    log_q = log_q[ind_L]
                log_p = model.target_distribution.log_prob(x)
                return torch.mean(log_q) - torch.mean(log_p)
            model.loss = modified_loss
        elif loss_type == 'flow_alpha_2_div_nis':
            def modified_loss(bs):
                x, log_q_x = model.flow.sample_and_log_prob((bs,))
                ind_L = filter_chirality(x)
                if torch.mean(1. * ind_L) > 0.1:
                    x = x[ind_L, :]
                    log_q_x = log_q_x[ind_L]
                log_p_x = model.target_distribution.log_prob(x)
                loss = - torch.mean(torch.exp(2 * (log_p_x - log_q_x)).detach() * log_q_x)
                return loss
            model.loss = modified_loss

# Start training
start_time = time()

for it in range(start_iter, max_iter):
    # Get loss
    if loss_type == 'flow_forward_kl' or lam_fkld is not None:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)
        x = x.to(device, non_blocking=True)
        if lam_fkld is None:
            loss = model.loss(x)
        else:
            loss = model.loss(batch_size) + lam_fkld * model.flow_forward_kl(x)
    elif use_rb:
        if rb_config['type'] == 'uniform':
            if it % rb_config['n_updates'] == 0:
                # Sample
                if transition_type == 'hmc':
                    x_ais, log_w_ais = model.annealed_importance_sampler.\
                        sample_and_log_weights(batch_size, logging=False)
                    x_ais = x_ais.detach()
                    log_w_ais = log_w_ais.detach()
                else:
                    with torch.no_grad():
                        x_ais, log_w_ais = model.annealed_importance_sampler.\
                            sample_and_log_weights(batch_size, logging=False)
                # Filter chirality
                if filter_chirality_train:
                    ind_L = filter_chirality(x_ais)
                    if torch.mean(1. * ind_L) > 0.1:
                        x_ais = x_ais[ind_L, :]
                        log_w_ais = log_w_ais[ind_L]
                # Optionally do clipping
                if rb_config['clip_w_frac'] is not None:
                    k = max(2, int(rb_config['clip_w_frac'] * log_w_ais.shape[0]))
                    max_log_w = torch.min(torch.topk(log_w_ais, k, dim=0).values)
                    log_w_ais = torch.clamp_max(log_w_ais, max_log_w)
                # Compute loss
                loss = model.fab_alpha_div_loss_inner(x_ais, log_w_ais)
                # Sample from buffer
                buffer_sample = buffer.sample_n_batches(batch_size=batch_size,
                                                        n_batches=rb_config['n_updates'] - 1)
                buffer_iter = iter(buffer_sample)
                # Add sample to buffer
                buffer.add(x_ais, log_w_ais)
            else:
                x, log_w = next(buffer_iter)
                loss = model.fab_alpha_div_loss_inner(x, log_w)
        elif rb_config['type'] == 'prioritised':
            if it % rb_config['n_updates'] == 0:
                # Sample
                if transition_type == 'hmc':
                    x_ais, log_w_ais = model.annealed_importance_sampler.\
                        sample_and_log_weights(batch_size, logging=False)
                    x_ais = x_ais.detach()
                    log_w_ais = log_w_ais.detach()
                else:
                    with torch.no_grad():
                        x_ais, log_w_ais = model.annealed_importance_sampler.\
                            sample_and_log_weights(batch_size, logging=False)
                # Filter chirality
                if filter_chirality_train:
                    ind_L = filter_chirality(x_ais)
                    if torch.mean(1. * ind_L) > 0.1:
                        x_ais = x_ais[ind_L, :]
                        log_w_ais = log_w_ais[ind_L]
                log_q_x = model.flow.log_prob(x_ais)
                # Add sample to buffer
                buffer.add(x_ais.detach(), log_w_ais.detach(), log_q_x.detach())
                # Sample from buffer
                buffer_sample = buffer.sample_n_batches(batch_size=batch_size,
                                                        n_batches=rb_config['n_updates'])
                buffer_iter = iter(buffer_sample)

            # Get batch from buffer
            x, log_w, log_q_old, indices = next(buffer_iter)
            x, log_w, log_q_old, indices = x.to(device), log_w.to(device), \
                                           log_q_old.to(device), indices.to(device)
            log_q_x = model.flow.log_prob(x)
            # Adjustment to account for change to theta since sample was last added/adjusted
            log_w_adjust = log_q_old - log_q_x.detach()
            w_adjust = torch.clip(torch.exp(log_w_adjust), max=rb_config['max_adjust_w_clip'])
            # Manually calculate the new form of the loss
            loss = - torch.mean(w_adjust * log_q_x)
            # Adjust buffer samples
            buffer.adjust(log_w_adjust, log_q_x.detach(), indices)

    else:
        loss = model.loss(batch_size)

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        if grad_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_grad_norm)
            grad_norm_append = np.array([[it + 1, grad_norm.item()]])
            grad_norm_hist = np.concatenate([grad_norm_hist,
                                             grad_norm_append])
        optimizer.step()

    # Update Lipschitz constant if flows are residual
    if flow_type == 'residual':
        nf.utils.update_lipschitz(model, 5)

    # Log loss
    loss_append = np.array([[it + 1, loss.item()]])
    loss_hist = np.concatenate([loss_hist, loss_append])

    # Clear gradients
    nf.utils.clear_grad(model)

    # Update lr scheduler
    if lr_scheduler is not None and (it + 1) % lr_step == 0:
        lr_scheduler.step()

    # Do lr warmup if needed
    if lr_warmup and it <= config['training']['warmup_iter']:
        warmup_scheduler.step()

    # Save loss
    if (it + 1) % log_iter == 0 or it == max_iter - 1:
        # Loss
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist,
                   delimiter=',', header='it,loss', comments='')
        # Gradient clipping
        if grad_clipping:
            np.savetxt(os.path.join(log_dir, 'grad_norm.csv'),
                       grad_norm_hist, delimiter=',',
                       header='it,grad_norm', comments='')

        # Disable step size tuning while evaluating model
        model.transition_operator.set_eval_mode(True)
        if use_rb and rb_config['type'] == 'prioritised':
            model.annealed_importance_sampler.target_log_prob = model.target_distribution.log_prob

        # Effective sample size
        if config['fab']['transition_type'] == 'hmc':
            base_samples, base_log_w, ais_samples, ais_log_w = \
                model.annealed_importance_sampler.generate_eval_data(8 * batch_size,
                                                                     batch_size)
        else:
            with torch.no_grad():
                base_samples, base_log_w, ais_samples, ais_log_w = \
                    model.annealed_importance_sampler.generate_eval_data(8 * batch_size,
                                                                         batch_size)
        # Re-enable step size tuning
        if config['fab']['adjust_step_size']:
            model.transition_operator.set_eval_mode(False)
        if use_rb and rb_config['type'] == 'prioritised':
            model.annealed_importance_sampler.target_log_prob = ais_target_log_prob

        ess_append = np.array([[it + 1, effective_sample_size(base_log_w, normalised=False),
                                effective_sample_size(ais_log_w, normalised=False)]])
        ess_hist = np.concatenate([ess_hist, ess_append])
        np.savetxt(os.path.join(log_dir, 'ess.csv'), ess_hist,
                   delimiter=',', header='it,flow,ais', comments='')
        if use_gpu:
            torch.cuda.empty_cache()

    if (it + 1) % checkpoint_iter == 0 or it == max_iter - 1:
        # Save checkpoint
        model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(cp_dir, 'optimizer.pt'))
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(),
                       os.path.join(cp_dir, 'lr_scheduler.pt'))
        if lr_warmup:
            torch.save(warmup_scheduler.state_dict(),
                       os.path.join(cp_dir, 'warmup_scheduler.pt'))

        # Disable step size tuning while evaluating model
        model.transition_operator.set_eval_mode(True)
        if use_rb and rb_config['type'] == 'prioritised':
            buffer.save(os.path.join(cp_dir, 'buffer.pt'))
            model.annealed_importance_sampler.target_log_prob = model.target_distribution.log_prob

        # Draw samples
        z_samples = torch.zeros(0, ndim).to(device)
        while z_samples.shape[0] < eval_samples_flow:
            with torch.no_grad():
                z_ = model.flow.sample((batch_size,))
            if filter_chirality_eval:
                ind_L = filter_chirality(z_)
                if torch.mean(1. * ind_L) > 0.1:
                    z_ = z_[ind_L, :]
            z_samples = torch.cat((z_samples, z_.detach()))
        z_samples = z_samples[:eval_samples_flow, :]

        # Evaluate model and save plots
        if 'snf' in config['flow']:
            log_prob_fn = lambda a: a.new_zeros(a.shape[0])
        else:
            log_prob_fn = model.flow.log_prob
        evaluate_aldp(z_samples, test_data, log_prob_fn,
                      target.coordinate_transform, it, metric_dir=log_dir_flow,
                      plot_dir=plot_dir_flow)

        # Draw samples
        z_samples = torch.zeros(0, ndim).to(device)
        while z_samples.shape[0] < eval_samples:
            if config['fab']['transition_type'] == 'hmc':
                z_ = model.annealed_importance_sampler.sample_and_log_weights(batch_size,
                                                                              logging=False)[0]
            else:
                with torch.no_grad():
                    z_ = model.annealed_importance_sampler.sample_and_log_weights(batch_size,
                                                                                  logging=False)[0]
            z_, _ = model.flow._nf_model.flows[-1].inverse(z_.detach())
            if filter_chirality_eval:
                ind_L = filter_chirality(z_)
                if torch.mean(1. * ind_L) > 0.1:
                    z_ = z_[ind_L, :]
            z_samples = torch.cat((z_samples, z_.detach()))

        # Evaluate model and save plots
        if eval_samples > 0:
            z_samples = z_samples[:eval_samples, :]
            evaluate_aldp(z_samples, test_data, log_prob_fn,
                          target.coordinate_transform, it, metric_dir=log_dir_ais,
                          plot_dir=plot_dir_ais)

        # Re-enable step size tuning
        if config['fab']['adjust_step_size']:
            model.transition_operator.set_eval_mode(False)
        if use_rb and rb_config['type'] == 'prioritised':
            model.annealed_importance_sampler.target_log_prob = ais_target_log_prob

    # End job if necessary
    if it % checkpoint_iter == 0 and args.tlimit is not None:
        time_past = (time() - start_time) / 3600
        num_cp = (it + 1 - start_iter) / checkpoint_iter
        if num_cp > .5 and time_past * (1 + 1 / num_cp) > args.tlimit:
            break
