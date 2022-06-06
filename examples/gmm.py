import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from fab.utils.plotting import plot_contours, plot_marginal_pair
from examples.setup_run import setup_trainer_and_run, Plotter
import torch


def setup_many_well_plotter(cfg: DictConfig, target, buffer=None) -> Plotter:
    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)

    def plot(fab_model, n_samples: int = cfg.training.batch_size):
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        plot_contours(target.log_prob, bounds=plotting_bounds, ax=axs[0], n_contour_levels=50,
                      grid_width_n_points=200)
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=axs[1], n_contour_levels=50,
                      grid_width_n_points=200)

        # plot flow samples
        samples_flow = fab_model.flow.sample((n_samples,))
        plot_marginal_pair(samples_flow, ax=axs[0], bounds=plotting_bounds)

        # plot ais samples
        samples_ais = fab_model.annealed_importance_sampler.sample_and_log_weights(n_samples,
                                                                                   logging=False)[0]
        plot_marginal_pair(samples_ais, ax=axs[1], bounds=plotting_bounds)



        axs[0].set_title("flow samples")
        axs[1].set_title("ais samples")
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            # plot buffer samples
            plot_contours(target.log_prob, bounds=plotting_bounds, ax=axs[2], n_contour_levels=50,
                          grid_width_n_points=200)
            samples_buffer = buffer.sample(n_samples)[0]
            plot_marginal_pair(samples_buffer, ax=axs[2], bounds=plotting_bounds)
            axs[2].set_title("buffer samples")
        plt.show()
        return [fig]
    return plot


def _run(cfg: DictConfig):
    from fab.target_distributions.gmm import GMM
    torch.manual_seed(cfg.training.seed)
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling)
    setup_trainer_and_run(cfg, setup_plotter=setup_many_well_plotter, target=target)


@hydra.main(config_path="./config/paper", config_name="gmm.yaml")
def run(cfg: DictConfig):
    _run(cfg)

if __name__ == '__main__':
    run()