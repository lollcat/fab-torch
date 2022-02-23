from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from fab.sampling_methods import AnnealedImportanceSampler, Metropolis, HamiltoneanMonteCarlo
from fab.utils.logging import ListLogger
from fab.target_distributions import TargetDistribution
from fab.target_distributions.gmm import GMM
from fab.wrappers.torch import WrappedTorchDist
from fab.utils.plotting import plot_history

def setup_ais(dim: int = 2,
            n_ais_intermediate_distributions: int = 40,
            seed: int = 0,
            transition_operator_type: str = "hmc",
             ) -> Tuple[AnnealedImportanceSampler, TargetDistribution]:
    # set up key objects
    torch.manual_seed(seed)
    target = GMM(dim=dim, n_mixes=4, loc_scaling=8)
    base_dist = WrappedTorchDist(torch.distributions.MultivariateNormal(loc=torch.zeros(dim),
                                                                 scale_tril=3*torch.eye(dim)))
    # setup transition operator
    if transition_operator_type == "hmc":
        transition_operator = HamiltoneanMonteCarlo(
            n_ais_intermediate_distributions=n_ais_intermediate_distributions,
            n_outer=5,
            epsilon=1.0, L=5, dim=dim,
            step_tuning_method="p_accept")
    elif transition_operator_type == "metropolis":
        transition_operator = Metropolis(n_transitions=n_ais_intermediate_distributions,
                                         n_updates=5)
    else:
        raise NotImplementedError
    ais = AnnealedImportanceSampler(base_distribution=base_dist,
                                    target_log_prob=target.log_prob,
                                    transition_operator=transition_operator,
                                    n_intermediate_distributions=n_ais_intermediate_distributions,
                                    )
    return ais, target


def test_ais__eval_batch(
        outer_batch_size: int = 100,
        inner_batch_size: int = 100,
        dim: int = 2,
        n_ais_intermediate_distributions: int = 40,
        seed: int = 0,
        transition_operator_type: str = "hmc",
             ):
    ais, _ = setup_ais(dim=dim, n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                    seed=seed, transition_operator_type=transition_operator_type)
    base_samples, base_log_w, ais_samples, ais_log_w = ais.generate_eval_data(outer_batch_size,
                                                                              inner_batch_size)
    assert base_samples.shape == (outer_batch_size, dim)
    assert ais_samples.shape == (outer_batch_size, dim)
    assert base_log_w.shape == (outer_batch_size,)
    assert ais_log_w.shape == (outer_batch_size, )



def test_ais__overall(dim: int = 2,
            n_ais_intermediate_distributions: int = 40,
            n_iterations: int = 40,
            batch_size: int = 1000,
            seed: int = 0,
            transition_operator_type: str = "hmc",
             ) -> None:
    ais, target = setup_ais(dim=dim,
                          n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                    seed=seed, transition_operator_type=transition_operator_type)
    logger = ListLogger()

    # set up plotting
    n_plots = 4
    n_plots_total = n_plots + 2
    fig, axs = plt.subplots(n_plots_total, figsize=(3, n_plots_total*3), sharex=True, sharey=True)
    plot_number_iterator = iter(range(n_plots_total))

    # plot base and target
    true_samples = target.sample((batch_size,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index].plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("target samples")

    sampler_samples = ais.base_distribution.sample((batch_size,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    plotting_iterations = list(np.linspace(0, n_iterations-1, n_plots, dtype="int"))
    axs[plot_index].plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    axs[plot_index].set_title("base samples")

    # estimate performance metrics over base distribution
    x, log_w = ais.base_distribution.sample_and_log_prob((batch_size,))
    performance_metrics = target.performance_metrics(x, log_w, n_batches_stat_aggregation=5)
    print(f"Performance metrics over base distribution {performance_metrics}")


    dim = 2
    dist = torch.distributions.MultivariateNormal(loc=torch.zeros(dim),
                                                                 scale_tril=3*torch.eye(dim))

    # run test
    for i in range(n_iterations):
        print(dist.sample((1,)))
        x, log_w = ais.sample_and_log_weights(batch_size=batch_size)
        performance_metrics = target.performance_metrics(x, log_w, n_batches_stat_aggregation=5)
        logging_info = ais.get_logging_info()
        logging_info.update(performance_metrics)
        logger.write(logging_info)

        # Plotting progress as the transition operator gets tuned.
        if i in plotting_iterations:
            x = x.cpu().detach()
            plot_index = next(plot_number_iterator)
            axs[plot_index].plot(x[:, 0], x[:, 1], "o", alpha=0.5)
            axs[plot_index].set_title(f"samples, iteration {i}")
            fig.show()

    plt.tight_layout()
    fig.show()

    plot_history(logger.history)
    plt.show()



