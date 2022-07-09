import os
from typing import Optional
import hydra
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from omegaconf import DictConfig
from examples.make_flow import make_wrapped_normflowdist
from examples.many_well_visualise_all_marginal_pairs import get_target_log_prob_marginal_pair
from fab.utils.plotting import plot_contours, plot_marginal_pair
from fab.target_distributions.many_well import ManyWellEnergy
import torch

PATH = os.getcwd()


def plot_marginals(cfg: DictConfig, supfig, model_name, plot_y_label):
    n_samples: int = 200
    alpha = 0.3
    torch.manual_seed(cfg.training.seed)
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)

    plotting_bounds = (-3, 3)

    dim = cfg.target.dim
    flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                     layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                     act_norm=cfg.flow.act_norm)

    if model_name:
        path_to_model = f"{PATH}/models/{model_name}_seed1.pt"
        checkpoint = torch.load(path_to_model, map_location="cpu")
        flow._nf_model.load_state_dict(checkpoint['flow'])

    samples_flow = flow.sample((n_samples,)).detach()

    axs = supfig.subplots(2, 2, sharex="row", sharey="row")

    for i in range(2):
        for j in range(2):
            target_log_prob = get_target_log_prob_marginal_pair(target.log_prob_2D, i, j+2)
            plot_contours(target_log_prob, bounds=plotting_bounds, ax=axs[i, j],
                          n_contour_levels=20)
            plot_marginal_pair(samples_flow, marginal_dims=(i, j+2),
                               ax=axs[i, j], bounds=plotting_bounds, alpha=alpha)


            if j == 0:
                if plot_y_label:
                    axs[i, j].set_ylabel(f"$x_{i + 1}$")
            if i == 1:
                axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")




@hydra.main(config_path="./", config_name="config.yaml")
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
    titles = ["FAB w/ buffer", "Flow w/ KLD"]

    width, height = 10, 6
    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    subfigs = fig.subfigures(1, 2, wspace=0.01)

    plot_marginals(cfg, subfigs[0], model_names[0], plot_y_label=True)
    subfigs[0].suptitle(titles[0])

    plot_marginals(cfg, subfigs[1], model_names[1], plot_y_label=False)
    subfigs[1].suptitle(titles[1])

    #fig.suptitle(' ', fontsize='xx-large')
    plt.show()


if __name__ == '__main__':
    run()