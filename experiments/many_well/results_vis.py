from typing import Optional
import os

import hydra
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from omegaconf import DictConfig
import torch

from fab.utils.plotting import plot_contours, plot_marginal_pair
from fab.target_distributions.many_well import ManyWellEnergy
from experiments.load_model_for_eval import load_model
from experiments.setup_run import setup_model
from experiments.many_well.many_well_visualise_all_marginal_pairs import get_target_log_prob_marginal_pair

PATH = os.getcwd()


def plot_manywell_results(cfg: DictConfig,
                          supfig,
                          path_to_model: Optional[str] = None,
                          plot_y_label: bool = True):
    """Visualise samples from marginal pair distributions for the first 4 dimensions of the
    Many Well problem."""
    n_samples: int = 200
    alpha = 0.3
    torch.manual_seed(cfg.training.seed)
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)

    plotting_bounds = (-3, 3)

    dim = cfg.target.dim

    if path_to_model:
        model = load_model(cfg, target, path_to_model)
    else:
        model = setup_model(cfg, target)

    samples_flow = model.flow.sample((n_samples,)).detach()

    axs = supfig.subplots(2, 2, sharex="row", sharey="row")

    for i in range(2):
        for j in range(2):
            # target_log_prob = get_target_log_prob_marginal_pair(target.log_prob_2D, i, j+2, dim)
            target_log_prob = get_target_log_prob_marginal_pair(target.log_prob, i, j + 2, dim)
            plot_contours(target_log_prob, bounds=plotting_bounds, ax=axs[i, j],
                          n_contour_levels=20, grid_width_n_points=100)
            plot_marginal_pair(samples_flow, marginal_dims=(i, j+2),
                               ax=axs[i, j], bounds=plotting_bounds, alpha=alpha)


            if j == 0:
                if plot_y_label:
                    axs[i, j].set_ylabel(f"$x_{i + 1}$")
            if i == 1:
                axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")




@hydra.main(config_path="../config", config_name="many_well.yaml")
def run(cfg: DictConfig):
    mpl.rcParams['figure.dpi'] = 300
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('figure', titlesize=15)
    rc('axes', titlesize=13, labelsize=13)  # fontsize of the axes title and labels
    #rc('legend', fontsize=6)
    rc('xtick', labelsize=11)
    rc('ytick', labelsize=11)

    model_names = ["fab_buffer", "flow_kld"]
    titles = ["FAB w/ buffer (ours)", "Flow w/ KLD"]
    seed = 1

    width, height = 10, 6
    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    subfigs = fig.subfigures(1, 2, wspace=0.01)

    path_to_model_0 = f"{PATH}/models/{model_names[0]}_seed{seed}.pt"
    plot_manywell_results(cfg, subfigs[0], path_to_model=path_to_model_0, plot_y_label=True)
    subfigs[0].suptitle(titles[0])

    path_to_model_1 = f"{PATH}/models/{model_names[1]}_seed{seed}.pt"
    plot_manywell_results(cfg, subfigs[1], path_to_model=path_to_model_1, plot_y_label=False)
    subfigs[1].suptitle(titles[1])

    #fig.suptitle(' ', fontsize='xx-large')
    plt.show()


if __name__ == '__main__':
    run()