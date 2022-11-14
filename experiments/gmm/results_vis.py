import os
from typing import Optional

import hydra
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from omegaconf import DictConfig
import torch
torch.set_default_dtype(torch.float64)
import numpy as np

from fab.utils.plotting import plot_contours, plot_marginal_pair
from experiments.load_model_for_eval import load_model
from experiments.setup_run import setup_model
from experiments.gmm.evaluation import setup_target

PATH = os.getcwd()
N_SAMPLES_PLOTTING = 1028

mpl.rcParams['figure.dpi'] = 300
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('axes', titlesize=20, labelsize=19)  # fontsize of the axes title and labels
rc('xtick', labelsize=17)
rc('ytick', labelsize=17)



def plot_result(cfg: DictConfig, ax: plt.axes, path_to_model: Optional[str], target):
    n_samples: int = N_SAMPLES_PLOTTING
    alpha = 0.3
    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)

    if path_to_model:
        model = load_model(cfg, target, path_to_model)
    else:
        model = setup_model(cfg, target)
    samples_flow = model.flow.sample((n_samples,)).detach()
    plot_marginal_pair(samples_flow, ax=ax, bounds=plotting_bounds, alpha=alpha)


@hydra.main(config_path="../config", config_name="gmm.yaml")
def run(cfg: DictConfig):
    seed = 0
    no_init = True
    if no_init:
        model_names = ["target_kld", "flow_nis",
                       "flow_kld", "rsb", "snf", "craft", "fab_no_buffer", "fab_buffer"]
        titles = ["Flow w/ ML",
                  r"Flow w/ $D_{\alpha=2}$",
                  "Flow w/ KLD", "RSB w/ KLD",
                  "SNF w/ KLD", "CRAFT", "FAB w/o buffer (ours)",
                  "FAB w/ buffer (ours)"]
    else:
        model_names = [None, "target_kld",
                       "flow_kld", "rsb", "snf", "craft",
                       "fab_no_buffer", "fab_buffer"]
        titles = ["Initialisation", "Flow w/ ML",
                  "Flow w/ KLD", "RBD w/ KLD",
                  "SNF w/ KLD",
                  "CRAFT",
                  "FAB w/o buffer (ours)",
                  "FAB w/ buffer (ours)"]

    if len(model_names) == 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 2, 4
    size = 3.2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*size, n_rows*size))
    axs[0, 0].set_ylabel(r"$x_2$")
    axs[1, 0].set_ylabel(r"$x_2$")
    for i in range(n_cols):
        axs[-1, i].set_xlabel(r"$x_1$")

    axs = axs.flatten()

    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)
    target = setup_target(cfg, num_samples=1)

    for i, (ax, model_name, title) in enumerate(zip(axs[:len(titles)], model_names, titles)):
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=ax, n_contour_levels=50,
                      grid_width_n_points=200)

        # Plot samples from model
        if model_name != "craft":
            if model_name and model_name[0:3] == "snf":
                # Update flow architecture for SNF if used.
                cfg.flow.use_snf = True
            else:
                cfg.flow.use_snf = False
            if model_name and model_name[0:3] == "rsb":
                cfg.flow.resampled_base = True
            else:
                cfg.flow.resampled_base = False
            path_to_model = f"{PATH}/models/{model_name}_seed{seed}.pt" if model_name else None
            plot_result(cfg, ax, path_to_model, target)
        else:
            # Plot craft.
            samples_craft = np.load(open(f"{PATH}/models/samples_craft_seed1.np", "rb"))
            samples_craft = torch.tensor(samples_craft)[:N_SAMPLES_PLOTTING]
            plot_marginal_pair(samples_craft, ax=ax, bounds=plotting_bounds, alpha=0.3)

        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if no_init:
        fig.savefig(f"{PATH}/plots/MoG_appendix.png", bbox_inches="tight")
    else:
        fig.savefig(f"{PATH}/plots/MoG.png", bbox_inches="tight")
    # fig.savefig(f"{PATH}/plots/MoG.png", bbox_inches="tight")
    plt.show()


@hydra.main(config_path="../config", config_name="gmm.yaml")
def plot_alpha_study(cfg: DictConfig):
    seed = 0
    alpha_values = [0.25,  0.5, 1.0, 1.5, 2.0, 3.0]
    titles = [r"$\alpha=0.25$", r"$\alpha=0.5$", r"$\alpha=1$", r"$\alpha=1.5$",
              r"$\alpha=2$", r"$\alpha=3$"]
    target = setup_target(cfg, num_samples=1)

    for fab_type in ["buff", "no_buff"]:
        n_rows, n_cols = 2, 3
        size = 3.2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*size, n_rows*size))
        axs[0, 0].set_ylabel(r"$x_2$")
        axs[1, 0].set_ylabel(r"$x_2$")
        for i in range(n_cols):
            axs[-1, i].set_xlabel(r"$x_1$")

        axs = axs.flatten()

        plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)
        for i, (ax, alpha, title) in enumerate(zip(axs[:len(titles)], alpha_values, titles)):
            plot_contours(target.log_prob, bounds=plotting_bounds, ax=ax, n_contour_levels=50,
                          grid_width_n_points=200)
            # Plot samples from model
            name_without_seed = f"{fab_type}_alpha{alpha}"
            name = name_without_seed + f"_seed{seed}"
            path_to_model = f"{PATH}/models_alpha/{name}.pt"
            plot_result(cfg, ax, path_to_model, target)

            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(f"{PATH}/plots/alpha_study_MoG_{fab_type}.png", bbox_inches="tight")
        plt.show()



if __name__ == '__main__':
    # run()
    fig = plot_alpha_study()
