import torch
from omegaconf import DictConfig
from matplotlib import pyplot as plt

from fab.utils.plotting import plot_marginal_pair, plot_contours
from experiments.load_model_for_eval import load_model


def get_target_log_prob_marginal_pair_alt(log_prob_2d, i: int, j: int):
    def log_prob(x):
        if i % 2 == 0:
            first_dim_x = torch.zeros_like(x)
            first_dim_x[:, 0] = x[:, 0]
        else:
            first_dim_x = torch.zeros_like(x)
            first_dim_x[:, 1] = x[:, 0]
        if j % 2 == 0:
            second_dim_x = torch.zeros_like(x)
            second_dim_x[:, 0] = x[:, 1]
        else:
            second_dim_x = torch.zeros_like(x)
            second_dim_x[:, 1] = x[:, 1]
        return log_prob_2d(first_dim_x) + log_prob_2d(second_dim_x)
    return log_prob

def get_target_log_prob_marginal_pair(log_prob, i: int, j: int, total_dim: int):
    def log_prob_marginal_pair(x_2d):
        x = torch.zeros((x_2d.shape[0], total_dim))
        x[:, i] = x_2d[:, 0]
        x[:, j] = x_2d[:, 1]
        return log_prob(x)
    return log_prob_marginal_pair


def plot_all_marginal_pairs(fab_model, dim, n_samples=500, plotting_bounds=(-3, 3)):
    """Plot all marginal pairs for a model."""
    n_rows = dim // 2
    fig, axs = plt.subplots(dim // 2, dim // 2, sharex=True, sharey=True, figsize=(n_rows * 3, n_rows * 3))

    samples_flow = fab_model.flow.sample((n_samples,))

    for i in range(n_rows):
        for j in range(n_rows):
            if i != j:
                log_prob_target = get_target_log_prob_marginal_pair(
                    fab_model.target_distribution.log_prob_2D, i, j, dim)
                plot_contours(log_prob_target, bounds=plotting_bounds, ax=axs[i, j], grid_width_n_points=40)
                plot_marginal_pair(samples_flow, ax=axs[i, j], marginal_dims=(i, j), bounds=plotting_bounds, alpha=0.2)

            if j == 0:
                axs[i, j].set_xlabel(f"dim {i}")
            if i == dim - 1:
                axs[i, j].set_xlabel(f"dim {j}")
    plt.show()


if __name__ == '__main__':
    # copy and paste config values, or load from `.yaml` file
    cfg = DictConfig({'target': {'dim': 32}, 'flow': {'layer_nodes_per_dim': 10, 'n_layers': 10}, 'fab':
        {'loss_type': 'alpha_2_div', 'transition_operator': {'type': 'hmc', 'n_inner_steps': 5},
         'n_intermediate_distributions': 4}, 'training':
        {'seed': 0, 'lr': 0.0001, 'batch_size': 512, 'n_iterations': 20000, 'use_gpu': True, 'use_64_bit': True,
         'use_buffer': True, 'prioritised_buffer': False, 'buffer_temp': 0.0, 'n_batches_buffer_sampling': 8,
         'maximum_buffer_length': 512000, 'min_buffer_length': 12560, 'log_w_clip_frac': None,
         'max_grad_norm': 10}, 'evaluation': {'n_plots': 100, 'n_eval': 200, 'eval_batch_size': 2560,
         'n_checkpoints': 10, 'save_path': 'results/many_well32/'}})
    model = load_model(cfg, "./models/model.pt")
    plot_all_marginal_pairs(model, cfg.target.dim)


