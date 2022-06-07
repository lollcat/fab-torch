import os
from typing import Optional
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from examples.make_flow import make_wrapped_normflowdist
from fab.utils.plotting import plot_contours, plot_marginal_pair
from fab.target_distributions.gmm import GMM
import torch

PATH = os.getcwd()

def plot_result(cfg: DictConfig, ax: plt.axes, model_name: Optional[str] = None):
    n_samples: int = 500
    alpha = 0.3
    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)

    dim = cfg.target.dim
    flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                     layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim)

    if model_name:
        path_to_model = f"{PATH}/models/{model_name}_model.pt"
        checkpoint = torch.load(path_to_model, map_location="cpu")
        flow._nf_model.load_state_dict(checkpoint['flow'])

    samples_flow = flow.sample((n_samples,)).detach()

    plot_marginal_pair(samples_flow, ax=ax, bounds=plotting_bounds, alpha=alpha)


@hydra.main(config_path="./", config_name="config.yaml")
def run(cfg: DictConfig):
    model_names = [None, "fab_no_buffer", "kld", "nis"]
    titles = ["Initialisation", "fab no buffer", "KLD over flow", r"$D_{\alpha=2}(p || q)$ over flow"]

    fig, axs = plt.subplots(2, 2, figsize=(8, 10))
    axs = axs.flatten()

    plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)
    torch.manual_seed(cfg.training.seed)
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling,
                 use_gpu=False)

    for ax, model_name, title in zip(axs, model_names, titles):
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=ax, n_contour_levels=50,
                      grid_width_n_points=200)
        plot_result(cfg, ax, model_name)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()