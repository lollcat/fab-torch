import torch
from omegaconf import DictConfig


from fab import FABModel, HamiltonianMonteCarlo, Metropolis
from fab.utils.plotting import plot_marginal_pair, plot_contours
from examples.make_flow import make_wrapped_normflowdist
from matplotlib import pyplot as plt


def setup_model(cfg: DictConfig, model_path):
    dim = cfg.target.dim  # applies to flow and target
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.training.seed)
    from fab.target_distributions.many_well import ManyWellEnergy
    assert dim % 2 == 0
    target = ManyWellEnergy(dim, a=-0.5, b=-6)
    plotting_bounds = (-3, 3)

    flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                     layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim)


    if cfg.fab.transition_operator.type == "hmc":
        # very lightweight HMC.
        transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            n_outer=1,
            epsilon=1.0, L=cfg.fab.transition_operator.n_inner_steps, dim=dim,
            step_tuning_method="p_accept")
    elif cfg.fab.transition_operator.type == "metropolis":
        transition_operator = Metropolis(n_transitions=cfg.fab.n_intermediate_distributions,
                                         n_updates=cfg.transition_operator.n_inner_steps,
                                         adjust_step_size=True)
    else:
        raise NotImplementedError


    # use GPU if available
    if torch.cuda.is_available() and cfg.training.use_gpu:
      flow.cuda()
      transition_operator.cuda()
      print("utilising GPU")


    fab_model = FABModel(flow=flow,
                         target_distribution=target,
                         n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                         transition_operator=transition_operator,
                         loss_type=cfg.fab.loss_type)
    fab_model.load(model_path, "cpu")
    return fab_model

def get_target_log_prob_marginal_pair(log_prob_2d, i, j):
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


def plot_marginal_pairs(fab_model, dim, n_samples=500, plotting_bounds=(-3,3)):
    n_rows = dim // 2
    fig, axs = plt.subplots(dim // 2, dim // 2, sharex=True, sharey=True, figsize=(n_rows * 3, n_rows * 3))

    samples_flow = fab_model.flow.sample((n_samples,))

    for i in range(n_rows):
        for j in range(n_rows):
            if i != j:
                log_prob_target = get_target_log_prob_marginal_pair(fab_model.target_distribution.log_prob_2D, i, j)
                plot_contours(log_prob_target, bounds=plotting_bounds, ax=axs[i, j])
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
    model = setup_model(cfg, "./models/model.pt")
    plot_marginal_pairs(model, cfg.target.dim)


