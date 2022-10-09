# Import modules
import argparse
import os
import torch
import numpy as np

import boltzgen as bg

from fab.utils.training import load_config
from experiments.make_flow.make_aldp_model import make_aldp_model



# Parse input arguments
parser = argparse.ArgumentParser(description='Sample from Boltzmann Generator')

parser.add_argument('--config', type=str, default='../config/bm.yaml',
                    help='Path config file specifying model '
                         'architecture')
parser.add_argument('--config_ais', type=str, default=None,
                    help='Path config file specifying parameters for '
                         'annealed importance sampling')
parser.add_argument('--mode', type=str, default='gpu',
                    help='Compute mode, can be cpu, or gpu')
parser.add_argument('--precision', type=str, default='double',
                    help='Precision to be used for computation, '
                         'can be float, double, or mixed')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed to be used for sampling')
parser.add_argument('--n_samples', type=int, default=1000000,
                    help='Number of samples to be drawn '
                         'from the base model')
parser.add_argument('--n_ais_samples', type=int, default=-1,
                    help='Number of samples to be drawn with '
                         'AIS, if -1 same number of samples '
                         'as from base model are drawn')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Batch size to be used for sampling, '
                         'if None the one from config is used')

args = parser.parse_args()

# Load config
config = load_config(args.config)

# Load AIS config if necessary
if args.config_ais is not None:
    config_ais = load_config(args.config_ais)
    config['fab'] = config_ais['fab']

# Precision
if args.precision == 'double':
    torch.set_default_dtype(torch.float64)

# GPU usage
use_gpu = not args.mode == 'cpu' and torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


# Set up model
model = make_aldp_model(config, device)

# Get checkpoint
root = config['training']['save_root']
cp_dir = os.path.join(root, 'checkpoints')
latest_cp = bg.utils.get_latest_checkpoint(cp_dir, 'model')

# Load checkpoint
model.load(latest_cp)
model.transition_operator.set_eval_mode(True)


# Sampling
if args.batch_size is None:
    batch_size = config['training']['batch_size']
else:
    batch_size = args.batch_size

seed = config['training']['seed'] if args.seed is None else args.seed

n_samples = args.n_samples
n_ais_samples = args.n_ais_samples
n_ais_samples = n_samples if n_ais_samples == -1 else n_ais_samples

s_dir = os.path.join(root, 'samples')
if not os.path.isdir(s_dir):
    os.makedirs(s_dir, exist_ok=True)

# Sample from flow model
if n_samples > 0:
    samples = np.zeros((n_samples, 60), dtype=args.precision)
    log_q = np.zeros(n_samples, dtype=args.precision)
    log_p = np.zeros(n_samples, dtype=args.precision)

    n_batches = int(np.ceil(n_samples / batch_size))

    torch.manual_seed(seed)

    # Draw samples
    for i in range(n_batches):
        if i == n_batches - 1:
            end = n_samples
            end_ = n_samples % batch_size
        else:
            end = (i + 1) * batch_size
            end_ = batch_size
        with torch.no_grad():
            z, lq = model.flow.sample_and_log_prob((batch_size,))
            lp = model.target_distribution.log_prob(z[:end_, :])
        samples[(i * batch_size):end, :] = z[:end_, :].detach().cpu().numpy()
        log_q[(i * batch_size):end] = lq[:end_].detach().cpu().numpy()
        log_p[(i * batch_size):end] = lp.detach().cpu().numpy()

    # Save samples
    path = os.path.join(s_dir, 'flow_samples_%03i.npz' % seed)
    np.savez_compressed(path, samples=samples, log_q=log_q, log_p=log_p)

del samples, log_q, log_p

# Do AIS
if n_ais_samples > 0:
    samples = np.zeros((n_ais_samples, 60), dtype=args.precision)
    log_w = np.zeros(n_ais_samples, dtype=args.precision)
    log_p = np.zeros(n_ais_samples, dtype=args.precision)

    n_batches = int(np.ceil(n_ais_samples / batch_size))

    torch.manual_seed(seed)

    # Draw samples
    for i in range(n_batches):
        if i == n_batches - 1:
            end = n_ais_samples
            end_ = n_ais_samples % batch_size
        else:
            end = (i + 1) * batch_size
            end_ = batch_size
        point, lw = model.annealed_importance_sampler.sample_and_log_weights(batch_size)
        z = point.x
        lp = model.target_distribution.log_prob(z[:end_, :].detach())
        samples[(i * batch_size):end, :] = z[:end_, :].detach().cpu().numpy()
        log_w[(i * batch_size):end] = lw[:end_].detach().cpu().numpy()
        log_p[(i * batch_size):end] = lp.detach().cpu().numpy()

    # Save samples
    path = os.path.join(s_dir, 'ais_samples_%03i.npz' % seed)
    np.savez_compressed(path, samples=samples, log_w=log_w, log_p=log_p)



