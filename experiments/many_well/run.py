import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from fab.utils.plotting import plot_contours, plot_marginal_pair
from experiments.setup_run import setup_trainer_and_run_flow, Plotter
import torch


def setup_many_well_plotter(cfg: DictConfig, target, buffer=None) -> Plotter:
    plotting_bounds = (-3, 3)

    def plot(fab_model, n_samples: int = cfg.training.batch_size, dim: int = cfg.target.dim):
        n_rows = dim // 2
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            fig, axs = plt.subplots(dim // 2, 3, sharex=True, sharey=True, figsize=(10, n_rows * 3))
        else:
            fig, axs = plt.subplots(dim // 2, 2, sharex=True, sharey=True, figsize=(10, n_rows * 3))

        samples_flow = fab_model.flow.sample((n_samples,)).detach()
        samples_ais = fab_model.annealed_importance_sampler.sample_and_log_weights(
            n_samples, logging=False)[0].x.detach()
        if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
            samples_buffer = buffer.sample(n_samples)[0].detach()

        for i in range(n_rows):
            plot_contours(target.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 0])
            plot_contours(target.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 1])

            # plot flow samples
            plot_marginal_pair(samples_flow, ax=axs[i, 0], bounds=plotting_bounds,
                               marginal_dims=(i * 2, i * 2 + 1))
            axs[i, 0].set_xlabel(f"dim {i * 2}")
            axs[i, 0].set_ylabel(f"dim {i * 2 + 1}")

            # plot ais samples
            plot_marginal_pair(samples_ais, ax=axs[i, 1], bounds=plotting_bounds,
                               marginal_dims=(i * 2, i * 2 + 1))
            axs[i, 1].set_xlabel(f"dim {i * 2}")
            axs[i, 1].set_ylabel(f"dim {i * 2 + 1}")

            if cfg.training.prioritised_buffer is True and cfg.training.use_buffer is True:
                plot_contours(target.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 2])
                plot_marginal_pair(samples_buffer, ax=axs[i, 2], bounds=plotting_bounds,
                                   marginal_dims=(i * 2, i * 2 + 1))
                axs[i, 2].set_xlabel(f"dim {i * 2}")
                axs[i, 2].set_ylabel(f"dim {i * 2 + 1}")

            plt.tight_layout()
        axs[0, 1].set_title("ais samples")
        axs[0, 0].set_title("flow samples")
        if cfg.training.use_buffer is True:
            axs[0, 2].set_title("buffer samples")
        plt.show()
        return [fig]
    return plot


def _run(cfg: DictConfig):
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.training.seed)
    from fab.target_distributions.many_well import ManyWellEnergy
    assert cfg.target.dim % 2 == 0
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=cfg.training.use_gpu)
    setup_trainer_and_run_flow(cfg, setup_many_well_plotter, target)


@hydra.main(config_path="../config", config_name="many_well.yaml")
def run(cfg: DictConfig):
    _run(cfg)

if __name__ == '__main__':
    run()
