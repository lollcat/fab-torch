import os
from typing import Optional
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from examples.make_flow import make_wrapped_normflowdist
from examples.setup_run_snf import make_normflow_snf_model, SNFModel
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
            path_to_model = f"{PATH}/models/{model_name}_model.pt"
            checkpoint = torch.load(path_to_model, map_location="cpu")
            snf.load_state_dict(checkpoint['flow'])
        # wrap appropriately
        snf = SNFModel(snf, cfg.target.dim)
        flow = snf.flow
    else:
        flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                         layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                         act_norm=cfg.flow.act_norm)

        if model_name:
            path_to_model = f"{PATH}/models/{model_name}_model.pt"
            checkpoint = torch.load(path_to_model, map_location="cpu")
            flow._nf_model.load_state_dict(checkpoint['flow'])

    samples_flow = flow.sample((n_samples,)).detach()

    plot_marginal_pair(samples_flow, ax=ax, bounds=plotting_bounds, alpha=alpha)


@hydra.main(config_path="./", config_name="config.yaml")
def run(cfg: DictConfig):
    model_names = [None, "fab_with_buffer", "fab_no_buffer", "flow_kld", "flow_nis", "snf"]
    titles = ["Initialisation", "fab with buffer", "fab no buffer",
              "KLD over flow", r"$D_{\alpha=2}(p || q)$ over flow", "SNF"]

    n_rows, n_cols = 2, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
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