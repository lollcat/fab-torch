import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from fab.utils.plotting import plot_contours, plot_marginal_pair
from experiments.setup_run import setup_trainer_and_run_flow, Plotter
from fab.target_distributions.gmm import GMM
import torch


def setup_gmm_plotter(cfg: DictConfig, target: GMM, buffer=None) -> Plotter:
    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)

    def plot(fab_model, n_samples: int = 800):
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        target.to("cpu")
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=axs[0], n_contour_levels=50,
                      grid_width_n_points=200)
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=axs[1], n_contour_levels=50,
                      grid_width_n_points=200)
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            plot_contours(target.log_prob, bounds=plotting_bounds, ax=axs[2], n_contour_levels=50,
                          grid_width_n_points=200)
        target.to(target.device)

        # plot flow samples
        samples_flow = fab_model.flow.sample((n_samples,)).detach()
        plot_marginal_pair(samples_flow, ax=axs[0], bounds=plotting_bounds)

        # plot ais samples
        samples_ais = fab_model.annealed_importance_sampler.sample_and_log_weights(n_samples,
                                                                                   logging=False)[0].x
        samples_ais = samples_ais.detach()
        plot_marginal_pair(samples_ais, ax=axs[1], bounds=plotting_bounds)


        axs[0].set_title("flow samples")
        axs[1].set_title("ais samples")
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            # plot buffer samples
            samples_buffer = buffer.sample(n_samples)[0].detach()
            plot_marginal_pair(samples_buffer, ax=axs[2], bounds=plotting_bounds)
            axs[2].set_title("buffer samples")
        # plt.show()
        return [fig]
    return plot


def _run(cfg: DictConfig):
    torch.manual_seed(0)  # seed of 0 for GMM problem
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling,
                 use_gpu=cfg.training.use_gpu)
    torch.manual_seed(cfg.training.seed)
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    setup_trainer_and_run_flow(cfg, setup_gmm_plotter, target)


@hydra.main(config_path="../config/", config_name="gmm.yaml")
def run(cfg: DictConfig):
    _run(cfg)

if __name__ == '__main__':
    run()
