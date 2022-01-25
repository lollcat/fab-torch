# Import modules
import argparse
import os
import torch
import numpy as np

import normflow as nf

from time import time
from fab.utils.training import load_config
from fab.target_distributions import AldpBoltzmann
from fab import FABModel
from fab.wrappers.normflow import WrappedNormFlowModel
from fab.sampling_methods.transition_operators import HamiltoneanMonteCarlo, Metropolis
from fab.utils.aldp import evaluateAldp

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
if 'seed' in config['training'] and config['training']['seed'] is not None:
    torch.manual_seed(config['training']['seed'])

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


# Set up model
# Flow
flow_type = config['flow']['type']
ndim = 60
# Flow layers
layers = []
for i in range(config['flow']['blocks']):
    if flow_type == 'rnvp':
        # Coupling layer
        hl = config['flow']['hidden_layers'] * [config['flow']['hidden_units']]
        scale_map = config['flow']['scale_map']
        scale = scale_map is not None
        param_map = nf.nets.MLP([(ndim + 1) // 2] + hl + [(ndim // 2) * (2 if scale else 1)],
                                init_zeros=config['flow']['init_zeros'])
        layers.append(nf.flows.AffineCouplingBlock(param_map, scale=scale,
                                                   scale_map=scale_map))
    else:
        raise NotImplementedError('The flow type ' + flow_type + ' is not implemented.')

    if config['flow']['mixing'] == 'affine':
        layers.append(nf.flows.InvertibleAffine(ndim, use_lu=True))
    elif config['flow']['mixing'] == 'permute':
        layers.append(nf.flows.Permute(ndim))

    if config['flow']['actnorm']:
        layers.append(nf.flows.ActNorm(ndim))
# Base distribution
if config['flow']['base']['type'] == 'gauss':
    base = nf.distributions.DiagGaussian(ndim,
                                         trainable=config['flow']['base']['learn_mean_var'])
else:
    raise NotImplementedError('The base distribution ' + config['flow']['base']['type']
                              + ' is not implemented')
flow = nf.NormalizingFlow(base, layers)
wrapped_flow = WrappedNormFlowModel(flow).to(device)

# Transition operator
if config['fab']['transition_type'] == 'hmc':
    # very lightweight HMC.
    transition_operator = HamiltoneanMonteCarlo(
        n_ais_intermediate_distributions=config['fab']['n_int_dist'], dim=ndim)
elif config['fab']['transition_type'] == 'metropolis':
    transition_operator = Metropolis(n_transitions=config['fab']['n_int_dist'],
                                     n_updates=5)
else:
    raise NotImplementedError('The transition operator ' + config['fab']['transition_type']
                              + ' is not implemented')
transition_operator = transition_operator.to(device)

# Target distribution
target = AldpBoltzmann(data_path=config['data']['transform'],
                       temperature=config['system']['temperature'],
                       energy_cut=config['system']['energy_cut'],
                       energy_max=config['system']['energy_max'],
                       n_threads=config['system']['n_threads'])
target = target.to(device)

# FAB model
model = FABModel(flow=wrapped_flow,
                 target_distribution=target,
                 n_intermediate_distributions=config['fab']['n_int_dist'],
                 transition_operator=transition_operator)

# Prepare output directories
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
plot_dir = os.path.join(root, 'plots')
log_dir = os.path.join(root, 'log')
# Create dirs if not existent
for dir in [cp_dir, plot_dir, log_dir]:
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
lr_warmup = 'warmup_iter' in config['training'] \
            and config['training']['warmup_iter'] is not None
if lr_warmup:
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda s: min(1., s / config['training']['warmup_iter']))
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=config['training']['rate_decay'])


# Train model
max_iter = config['training']['max_iter']
log_iter = config['training']['log_iter']
checkpoint_iter = config['training']['checkpoint_iter']
start_iter = 0

batch_size = config['training']['batch_size']
loss_hist = np.zeros((0, 2))
eval_samples = config['training']['eval_samples']
eval_batches = (eval_samples - 1) // batch_size + 1

# Start training
start_time = time()

for it in range(start_iter, max_iter):
    # Get loss
    loss = model.loss(batch_size)

    # Make step
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        optimizer.step()

    # Update Lipschitz constant if flows are residual
    if flow_type == 'residual':
        nf.utils.update_lipschitz(model, 5)

    # Log loss
    loss_append = np.array([[it + 1, loss.item()]])
    loss_hist = np.concatenate([loss_hist, loss_append])

    # Clear gradients
    nf.utils.clear_grad(model)

    # Do lr warmup if needed
    if lr_warmup and it <= config['training']['warmup_iter']:
        warmup_scheduler.step()

    # Update lr scheduler
    if (it + 1) % config['training']['decay_iter'] == 0:
        lr_scheduler.step()

    # Save loss
    if (it + 1) % log_iter == 0:
        np.savetxt(os.path.join(log_dir, 'loss.csv'), loss_hist,
                   delimiter=',', header='it,loss', comments='')
        if use_gpu:
            torch.cuda.empty_cache()

    if (it + 1) % checkpoint_iter == 0:
        # Save checkpoint
        """
        model.save(os.path.join(cp_dir, 'model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(cp_dir, 'optimizer.pt'))
        if lr_warmup:
            torch.save(warmup_scheduler.state_dict(),
                       os.path.join(cp_dir, 'warmup_scheduler.pt'))
        """

        # Draw samples
        z_samples = torch.zeros(0, 1).to(device)
        for i in range(eval_batches):
            if i == eval_batches - 1:
                ns = ((eval_samples - 1) % batch_size) + 1
            else:
                ns = batch_size
            z_ = model.annealed_importance_sampler.sample_and_log_weights(ns,
                                                                          logging=False)[0]
            z_samples = torch.cat((z_samples, z_.detach()))

        # Evaluate model and save plots
        evaluateAldp(z_samples, test_data, model.flow.log_prob,
                     target.coordinate_transform, it, metric_dir=log_dir,
                     plot_dir=plot_dir)

    # End job if necessary
    if it % checkpoint_iter == 0 and args.tlimit is not None:
        time_past = (time() - start_time) / 3600
        num_cp = (it + 1 - start_iter) / checkpoint_iter
        if num_cp > .5 and time_past * (1 + 1 / num_cp) > args.tlimit:
            break
