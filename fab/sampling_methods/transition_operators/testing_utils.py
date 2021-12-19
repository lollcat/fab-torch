import torch

torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from tqdm import tqdm

from fab.target_distributions.gmm import MoG
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
    target = MoG(dim=dim, n_mixes=3, loc_scaling=8)
    learnt_sampler = torch.distributions.MultivariateNormal(loc=torch.zeros(dim),
                                                                 scale_tril=2*torch.eye(dim))
    n_intermediate_plots = 2
    n_plots = 4 + n_intermediate_plots
    fig, axs = plt.subplots(n_plots, figsize=(3, n_plots*3), sharex=True, sharey=True)
    plot_number_iterator = iter(range(n_plots))
    for i in tqdm(range(n_iterations)):
        x = learnt_sampler.sample_n(n_samples)  # initialise the chain
        for j in range(n_ais_intermediate_distributions):
            # here we just aim for the target distribution rather than interpolating between,
            # as we are just testing the transition operator, and not AIS.
            x = transition_operator.transition(x, target.log_prob, j)
        transition_operator_info = transition_operator.get_logging_info()
        logger.write(transition_operator_info)
        if i == 0 or i == n_iterations - 1 or i % int(n_iterations / (n_intermediate_plots + 1)) \
                == 0:
            x = x.cpu().detach()
            plot_index = next(plot_number_iterator)
            axs[plot_index].plot(x[:, 0], x[:, 1], "o", alpha=0.5)
            axs[plot_index].set_title(f"transition operator output samples, iteration {i}")
            fig.show()

    true_samples = target.sample((n_samples,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("true samples")

    sampler_samples = learnt_sampler.sample_n(n_samples).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("sampler samples")
    plt.tight_layout()
    fig.show()

    plot_history(logger.history)
    plt.show()

