from typing import Optional, Dict

import abc
import torch
from fab.types_ import LogProbFunc

class TargetDistribution(abc.ABC):

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None) -> Dict:
        """
        Check performance metrics using samples & log weights from the model, as well as it's
        probability density function (if defined).
        Args:
            samples: Samples from the trained model.
            log_w: Log importance weights from the trained model.
            log_q_fn: Log probability density function of the trained model, if defined.

        Returns:
            info: A dictionary of performance measures, specific to the defined
            target_distribution, that evaluate how well the trained model approximates the target.
        """
        raise NotImplementedError


    def sample(self, shape):
        raise NotImplementedError
