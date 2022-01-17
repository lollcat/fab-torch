from typing import Optional, Dict

import abc
import torch

class TargetDistribution(abc.ABC):

    @abc.abstractmethod
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples, log_w, n_batches_stat_aggregation: Optional[int]) -> \
            Dict:
        """Check performance metrics using samples and log weights from the model. In cases where
        n_batches_stat_aggregation is included, the samples and weights are divided into
        n_batches_stat_aggregation mini-batches. Returns a dictionary of info."""
        raise NotImplementedError
