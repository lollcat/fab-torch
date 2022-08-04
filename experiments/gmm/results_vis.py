import os
from typing import Optional
import hydra
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from omegaconf import DictConfig
from experiments.make_flow import make_wrapped_normflowdist
from experiments.setup_run_snf import make_normflow_snf_model, SNFModel
from fab.utils.plotting import plot_contours, plot_marginal_pair
from fab.target_distributions.gmm import GMM
import torch

PATH = os.getcwd()

def plot_result(cfg: DictConfig, ax: plt.axes, target, model_name: Optional[str] = None):
    n_samples: int = 800
    alpha = 0.3
    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)

    dim = cfg.target.dim
    if model_name and model_name[0:3] == "snf":
        snf = make_normflow_snf_model(dim,
                                       n_flow_layers=cfg.flow.n_layers,
                                       layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                       act_norm=cfg.flow.act_norm,
                                       target=target
                                       )
        if model_name:
            path_to_model = f"{PATH}/models/{model_name}_seed1.pt"
            checkpoint = torch.load(path_to_model, map_location="cpu")
            snf.load_state_dict(checkpoint['flow'])
        # wrap appropriately
        snf = SNFModel(snf, target, cfg.target.dim)
        flow = snf.flow
    else:
        flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                         layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                         act_norm=cfg.flow.act_norm)

        if model_name:
            path_to_model = f"{PATH}/models/{model_name}_seed1.pt"
            checkpoint = torch.load(path_to_model, map_location="cpu")
            flow._nf_model.load_state_dict(checkpoint['flow'])

    samples_flow = flow.sample((n_samples,)).detach()

    plot_marginal_pair(samples_flow, ax=ax, bounds=plotting_bounds, alpha=alpha)


@hydra.main(config_path="../config", config_name="gmm.yaml")
def run(cfg: DictConfig):
    appendix = True
    if appendix:
        model_names = ["target_kld", "flow_nis", "flow_kld", "snf", "fab_no_buffer", "fab_buffer"]
        titles = ["Flow w/ ML", r"Flow w/ $D_{\alpha=2}$", "Flow w/ KLD",
                  "SNF w/ KLD", "FAB w/o buffer (ours)", "FAB w/ buffer (ours)"]
    else:
        model_names = [None, "target_kld", "flow_kld", "snf", "fab_no_buffer", "fab_buffer"]
        titles = ["Initialisation", "Flow w/ ML", "Flow w/ KLD",
                  "SNF w/ KLD", "FAB w/o buffer (ours)", "FAB w/ buffer (ours)"]
    mpl.rcParams['figure.dpi'] = 300
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('axes', titlesize=15, labelsize=13)  # fontsize of the axes title and labels
    #rc('legend', fontsize=6)
    rc('xtick', labelsize=11)
    rc('ytick', labelsize=11)

    n_rows, n_cols = 2, 3
    size = 3.2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*size, n_rows*size))
    axs[0, 0].set_ylabel(r"$x_2$")
    axs[1, 0].set_ylabel(r"$x_2$")
    for i in range(n_cols):
        axs[-1, i].set_xlabel(r"$x_1$")

    axs = axs.flatten()

    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)
    torch.manual_seed(cfg.training.seed)
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling,
                 use_gpu=False)
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)

    for i, (ax, model_name, title) in enumerate(zip(axs[:len(titles)], model_names, titles)):
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=ax, n_contour_levels=50,
                      grid_width_n_points=200)
        plot_result(cfg, ax, target, model_name)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    run()