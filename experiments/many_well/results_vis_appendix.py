import os

import hydra
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from omegaconf import DictConfig
import torch

from experiments.many_well.results_vis import plot_manywell_results


PATH = os.getcwd()



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

    torch.set_default_dtype(torch.float64)
    model_names = ["target_kld",
                   "flow_nis",
                   "flow_kld",
                   "snf_hmc",
                   "fab_no_buffer",
                   "fab_buffer"]
    titles = ["Flow w/ ML",
              r"Flow w/ $D_{\alpha=2}$",
              "Flow w/ KLD",
              "SNF w/ KLD",
              "FAB w/o buffer (ours)",
              "FAB w/ buffer (ours)"]

    width, height = 10, 15
    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    subfigs = fig.subfigures(3, 2, wspace=0.01).flatten()
    seed = 1

    for i, (ax, model_name, title) in enumerate(zip(subfigs[:len(titles)], model_names, titles)):
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
        plot_manywell_results(cfg, subfigs[i], path_to_model=path_to_model, plot_y_label=True)
        ax.suptitle(title)
    fig.savefig(f"{PATH}/plots/many_well_appendix.png", bbox_inches="tight")
    plt.show()

    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    subfig = fig.subfigures(1, wspace=0.01)
    model_name = "rbd"
    title = "RBD w/ KLD"
    cfg.flow.resampled_base = True
    path_to_model = f"{PATH}/models/{model_name}_seed{seed}.pt" if model_name else None
    plot_manywell_results(cfg, subfig, path_to_model=path_to_model, plot_y_label=True)
    subfig.suptitle(title)
    plt.savefig(f"{PATH}/plots/many_well_appendix_rbd.png", bbox_inches="tight")
    plt.show()




if __name__ == '__main__':
    run()