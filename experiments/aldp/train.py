# Import modules
import argparse
import os
import torch
import numpy as np

import normflows as nf
import boltzgen as bg

from time import time
from fab.utils.training import load_config
from fab.sampling_methods.transition_operators import HamiltonianMonteCarlo, Metropolis
from fab.utils.aldp import evaluate_aldp
from fab.utils.aldp import filter_chirality
from fab.utils.numerical import effective_sample_size
from fab.utils.replay_buffer import ReplayBuffer
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer
from fab.core import ALPHA_DIV_TARGET_LOSSES
from experiments.make_flow.make_aldp_model import make_aldp_model



# Parse input arguments
parser = argparse.ArgumentParser(description='Train Boltzmann Generator '
                                             'with various objectives')

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
model = make_aldp_model(config, device)

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

# Set parameters for training
ndim = 60
loss_type = 'fab_alpha_div' if 'loss_type' not in config['fab'] \
        else config['fab']['loss_type']
transition_type = config['fab']['transition_type']
flow_type = config['flow']['type']

# Load train data if needed
lam_fkld = None if not 'lam_fkld' in config['fab'] else config['fab']['lam_fkld']
if loss_type == 'forward_kl' or lam_fkld is not None:
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
            point, log_w = model.annealed_importance_sampler.sample_and_log_weights(
                batch_size, logging=False)
            return point.x, log_w
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
                point, log_w = model.annealed_importance_sampler.sample_and_log_weights(
                    batch_size, logging=False)
                return point.x, log_w, point.log_q

        buffer = PrioritisedReplayBuffer(dim=ndim, max_length=rb_config['max_length'] * batch_size,
                                         min_sample_length=rb_config['min_length'] * batch_size,
                                         initial_sampler=initial_sampler, device=str(device))

        if os.path.exists(buffer_path):
            buffer.load(buffer_path)
else:
    use_rb = False
    if filter_chirality_train:
        if loss_type == 'fab_alpha_div':
            def modified_loss(bs):
                point_ais, log_w_ais = model.annealed_importance_sampler.sample_and_log_weights(bs)
                log_w_ais = log_w_ais.detach()
                ind_L = filter_chirality(point_ais.x)
                if torch.mean(1. * ind_L) > 0.1:
                    point_ais = point_ais[ind_L]
                    log_w_ais = log_w_ais[ind_L]
                loss = model.fab_alpha_div_inner(point_ais, log_w_ais)
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

# Set AIS/transition operator target.
min_is_target = config['fab']['loss_type'] in ALPHA_DIV_TARGET_LOSSES
if 'replay_buffer' in config['training']:
    min_is_target = min_is_target or config['training']['replay_buffer']['type'] == 'prioritised'
alpha = None if not 'alpha' in config['fab'] else config['fab']['alpha']
model.set_ais_target(min_is_target=min_is_target)

# Start training
start_time = time()

for it in range(start_iter, max_iter):
    # Get loss
    if loss_type == 'forward_kl' or lam_fkld is not None:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)
        x = x.to(device, non_blocking=True)
        if lam_fkld is None:
            loss = model.loss(x)
        else:
            loss = model.loss(batch_size) + lam_fkld * model.forward_kl(x)
    elif use_rb:
        if rb_config['type'] == 'uniform':
            if it % rb_config['n_updates'] == 0:
                # Sample
                point_ais, log_w_ais = model.annealed_importance_sampler.\
                    sample_and_log_weights(batch_size, logging=False)
                # Filter chirality
                if filter_chirality_train:
                    ind_L = filter_chirality(point_ais.x)
                    if torch.mean(1. * ind_L) > 0.1:
                        point_ais = point_ais[ind_L, :]
                        log_w_ais = log_w_ais[ind_L]
                # Optionally do clipping
                if rb_config['clip_w_frac'] is not None:
                    k = max(2, int(rb_config['clip_w_frac'] * log_w_ais.shape[0]))
                    max_log_w = torch.min(torch.topk(log_w_ais, k, dim=0).values)
                    log_w_ais = torch.clamp_max(log_w_ais, max_log_w)
                # Compute loss
                loss = model.fab_ub_alpha_div_loss_inner(point_ais, log_w_ais)
                # Sample from buffer
                buffer_sample = buffer.sample_n_batches(batch_size=batch_size,
                                                        n_batches=rb_config['n_updates'] - 1)
                buffer_iter = iter(buffer_sample)
                # Add sample to buffer
                buffer.add(point_ais.x, log_w_ais)
            else:
                x, log_w = next(buffer_iter)
                log_q = model.flow.log_prob(x)
                log_p = model.target_distribution.log_prob(x)
                loss = model.fab_ub_alpha_div_loss_inner(log_q, log_p, log_w)
        elif rb_config['type'] == 'prioritised':
            if it % rb_config['n_updates'] == 0:
                # Sample
                point_ais, log_w_ais = model.annealed_importance_sampler.\
                    sample_and_log_weights(batch_size, logging=False)
                # Filter chirality
                if filter_chirality_train:
                    ind_L = filter_chirality(point_ais.x)
                    if torch.mean(1. * ind_L) > 0.1:
                        point_ais = point_ais[ind_L]
                        log_w_ais = log_w_ais[ind_L]
                # Add sample to buffer
                buffer.add(point_ais.x, log_w_ais.detach(), point_ais.log_q)
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
            log_w_adjust = (1 - alpha) * (log_q_x.detach() - log_q_old)
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
            model.set_ais_target(min_is_target=False)

        # Effective sample size.
        base_samples, base_log_w, ais_samples, ais_log_w = \
            model.annealed_importance_sampler.generate_eval_data(8 * batch_size,
                                                                 batch_size)
        # Re-enable step size tuning
        if config['fab']['adjust_step_size']:
            model.transition_operator.set_eval_mode(False)
        if use_rb and rb_config['type'] == 'prioritised':
            model.set_ais_target(min_is_target=True)

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
            model.set_ais_target(min_is_target=False)  # Eval over p and not p^2/q.

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
                      model.target_distribution.coordinate_transform, it, metric_dir=log_dir_flow,
                      plot_dir=plot_dir_flow)

        # Draw samples
        z_samples = torch.zeros(0, ndim).to(device)
        while z_samples.shape[0] < eval_samples:
            z_ = model.annealed_importance_sampler.sample_and_log_weights(batch_size,
                                                                              logging=False)[0].x
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
                          model.target_distribution.coordinate_transform, it, metric_dir=log_dir_ais,
                          plot_dir=plot_dir_ais)

        # Re-enable step size tuning
        if config['fab']['adjust_step_size']:
            model.transition_operator.set_eval_mode(False)
        if use_rb and rb_config['type'] == 'prioritised':
            model.set_ais_target(min_is_target=True)

    # End job if necessary
    if it % checkpoint_iter == 0 and args.tlimit is not None:
        time_past = (time() - start_time) / 3600
        num_cp = (it + 1 - start_iter) / checkpoint_iter
        if num_cp > .5 and time_past * (1 + 1 / num_cp) > args.tlimit:
            break
