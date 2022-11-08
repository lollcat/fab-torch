from typing import NamedTuple

import torch
import numpy as np


torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from tqdm import tqdm

from fab.target_distributions.gmm import GMM
from fab.sampling_methods.transition_operators import TransitionOperator
from fab.utils.plotting import plot_history
from fab.utils.logging import ListLogger
from fab.sampling_methods.base import create_point


class TransitionOperatorTestConfig(NamedTuple):
    p_target: bool = False
    alpha: float = 2.0
    n_ais_intermediate_distributions: int = 10
    beta_space = np.linspace(0, 1, n_ais_intermediate_distributions + 2)
    dim: int = 2
    seed: int = 0
    target_scale = 5.0
    base_scale: float = 1.0
    base_scale = base_scale * target_scale
    target = GMM(dim=dim, n_mixes=3, loc_scaling=target_scale)
    learnt_sampler = torch.distributions.MultivariateNormal(
        loc=torch.zeros(dim), scale_tril=base_scale * torch.eye(dim))


def test_transition_operator(
        transition_operator: TransitionOperator,
        config: TransitionOperatorTestConfig = TransitionOperatorTestConfig(),
        n_iterations: int = 20,
        n_samples: int = 1000,
        ) -> None:

    logger = ListLogger()
    torch.manual_seed(config.seed)
    # instantiate base and target distribution
    n_intermediate_plots = 4  # plots of samples over different HMC iterations
    n_plots = 2 + n_intermediate_plots
    fig, axs = plt.subplots(n_plots, figsize=(3, n_plots*3), sharex=True, sharey=True)
    plot_number_iterator = iter(range(n_plots))
    plot_iter = np.linspace(0, n_iterations-1, n_intermediate_plots, dtype="int")
    for i in tqdm(range(n_iterations)):
        x = config.learnt_sampler.sample((n_samples, ))  # initialise the chain
        point = create_point(x=x,
                             log_q_fn=config.learnt_sampler.log_prob,
                             log_p_fn=config.target.log_prob,
                             with_grad=transition_operator.uses_grad_info
                             )
        for j in range(config.n_ais_intermediate_distributions):
            # here we just aim for the target distribution rather than interpolating between,
            # as we are just testing the transition operator, and not AIS.
            point = transition_operator.transition(point, j + 1, config.beta_space[j+1])
        transition_operator_info = transition_operator.get_logging_info()
        logger.write(transition_operator_info)
        if i in plot_iter:
            x = point.x.cpu().detach()
            plot_index = next(plot_number_iterator)
            axs[plot_index].plot(x[:, 0], x[:, 1], "o", alpha=0.5)
            axs[plot_index].set_title(f"transition operator output samples, iteration {i}")
            fig.show()

    true_samples = config.target.sample((n_samples,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("true samples")

    sampler_samples = config.learnt_sampler.sample((n_samples,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("base dist samples")
    plt.tight_layout()
    fig.show()

    plot_history(logger.history)
    plt.show()

