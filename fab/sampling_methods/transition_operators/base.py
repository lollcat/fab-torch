from typing import Mapping, Any

import abc
import torch

from fab.types_ import LogProbFunc

class TransitionOperator(abc.ABC):

    def get_logging_info(self) -> Mapping[str, Any]:
        """Returns a dictionary of relevant information for logging."""
        raise NotImplementedError

    @abc.abstractmethod
    def transition(self, x: torch.Tensor, log_p_x: LogProbFunc, i: int) -> torch.Tensor:
        """
        Returns x generated from transition with log_q_x, as the invariant
        distribution.

        Args:
            x: Input samples from the base distribution if i = 0, else from the previous AIS step.
            log_p_x: Target probability density function, an interpolation between the base and
            target distributions.
            i: Step number in the sequence

        Returns:
            x: Samples from MCMC with log_p_x as the target distribution.
        """
        raise NotImplementedError
