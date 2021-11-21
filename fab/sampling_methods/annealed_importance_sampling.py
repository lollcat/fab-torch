from typing import Tuple

from fab.types import LogProbFunc
from fab.sampling_methods.transition_operators.base import TransitionOperator
import torch
# TODO: types

class AnnealedImportanceSampler:
    def __init__(self,
                 base_distribution,
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

    def sample_and_log_weights(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

