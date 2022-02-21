import torch
import numpy as np

torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from tqdm import tqdm

from fab.target_distributions.gmm import GMM
from fab.sampling_methods.transition_operators import TransitionOperator
from fab.utils.plotting import plot_history
from fab.utils.logging import ListLogger


def test_transition_operator(transition_operator: TransitionOperator,
                             n_ais_intermediate_distributions: int = 10,
                             n_iterations: int = 20,
                             n_samples: int = 1000,
                             dim: int = 2,
                             seed: int = 0) -> None:
    logger = ListLogger()
    torch.manual_seed(seed)
    # instantiate base and target distribution
    target = GMM(dim=dim, n_mixes=3, loc_scaling=6)
    learnt_sampler = torch.distributions.MultivariateNormal(loc=torch.zeros(dim),
                                                                 scale_tril=2*torch.eye(dim))
    n_intermediate_plots = 2
    n_plots = 4 + n_intermediate_plots
    fig, axs = plt.subplots(n_plots, figsize=(3, n_plots*3), sharex=True, sharey=True)
    plot_number_iterator = iter(range(n_plots))
    plot_iter = np.linspace(0, n_iterations-1, n_intermediate_plots, dtype="int")
    for i in tqdm(range(n_iterations)):
        x = learnt_sampler.sample((n_samples, ))  # initialise the chain
        for j in range(n_ais_intermediate_distributions):
            # here we just aim for the target distribution rather than interpolating between,
            # as we are just testing the transition operator, and not AIS.
            x = transition_operator.transition(x, target.log_prob, j)
        transition_operator_info = transition_operator.get_logging_info()
        logger.write(transition_operator_info)
        if i in plot_iter:
            x = x.cpu().detach()
            plot_index = next(plot_number_iterator)
            axs[plot_index].plot(x[:, 0], x[:, 1], "o", alpha=0.5)
            axs[plot_index].set_title(f"transition operator output samples, iteration {i}")
            fig.show()

    # TODO: fix plotting here
    true_samples = target.sample((n_samples,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("true samples")

    sampler_samples = learnt_sampler.sample((n_samples,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("base dist samples")
    plt.tight_layout()
    fig.show()

    plot_history(logger.history)
    plt.show()

