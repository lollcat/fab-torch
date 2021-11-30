from typing import Tuple

from fab.types import LogProbFunc
from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.learnt_distributions.base import LearntDistribubtion
import torch
import numpy as np


class AnnealedImportanceSampler:
    def __init__(self,
                 base_distribution: LearntDistribubtion,
                 target_log_prob: LogProbFunc,
                 transition_operator: TransitionOperator,
                 n_intermediate_distributions: int = 1,
                 distribution_spacing_type: str = "linear"
                 ):
        self.base_distribution = base_distribution
        self.target_log_prob = target_log_prob
        self.transition_operator = transition_operator
        self.n_intermediate_distributions = n_intermediate_distributions
        self.distribution_spacing_type = distribution_spacing_type
        self.B_space = self.setup_distribution_spacing(distribution_spacing_type,
                                                       n_intermediate_distributions)

    @property
    def device(self):
        return self.base_distribution.device


    def sample_and_log_weights(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_new, log_prob_p0 = self.base_distribution.sample_and_log_prob((batch_size,))
        log_w = self.intermediate_unnormalised_log_prob(x_new, 1) - log_prob_p0
        for j in range(1, self.n_intermediate_distributions+1):
            x_new, log_w = self.perform_transition(x_new, log_w, j)
        return x_new, log_w


    def perform_transition(self, x_new: torch.Tensor, log_w: torch.Tensor, j: int):
        target_p_x = lambda x: self.intermediate_unnormalised_log_prob(x, j)
        x_new = self.transition_operator.transition(x_new, target_p_x, j-1)
        log_w = log_w + self.intermediate_unnormalised_log_prob(x_new, j + 1) - \
                self.intermediate_unnormalised_log_prob(x_new, j)
        return x_new, log_w


    def intermediate_unnormalised_log_prob(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """Calculate the intermediate log probability density function, by interpolating between
        the base and target distributions log probability density functions."""
        # j is the step of the algorithm, and corresponds which
        # intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta) * self.base_distribution.log_prob(x) + beta * self.target_log_prob(x)


    def setup_distribution_spacing(self, distribution_spacing_type: str,
                                   n_intermediate_distributions: int) -> torch.Tensor:
        """Setup the spacing of the distributions, either with linear or geometric spacing."""
        assert n_intermediate_distributions > 0
        if n_intermediate_distributions < 3:
            print(f"using linear spacing as there is {n_intermediate_distributions}"
                  f"intermediate distribution")
            distribution_spacing_type = "linear"
        if distribution_spacing_type == "geometric":
            # rough heuristic, copying ratio used in example in AIS paper
            n_linspace_points = max(int(n_intermediate_distributions / 5), 2)
            n_geomspace_points = n_intermediate_distributions - n_linspace_points
            B_space = np.concatenate([np.linspace(0, 0.1, n_linspace_points + 1)[:-1],
                                   np.geomspace(0.1, 1, n_geomspace_points)])
        elif distribution_spacing_type == "linear":
            B_space = np.linspace(0.0, 1.0, n_intermediate_distributions+2)
        else:
            raise Exception(f"distribution spacing incorrectly specified:"
                            f" '{distribution_spacing_type}',"
                            f"options are 'geometric' or 'linear'")
        return torch.tensor(B_space)
