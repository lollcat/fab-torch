from typing import Mapping, Any
import torch

from fab.types_ import LogProbFunc
from fab.sampling_methods.base import Point

class TransitionOperator(torch.nn.Module):

    @property
    def uses_grad_info(self) -> bool:
        """Returns True if the transition operator uses gradient info (e.g. HMC) or False if not
        (e.g. Metropolis)."""
        raise NotImplementedError

    def get_logging_info(self) -> Mapping[str, Any]:
        """Returns a dictionary of relevant information for logging."""
        raise NotImplementedError


    def transition(self, x: Point, log_p_x: LogProbFunc, i: int) -> torch.Tensor:
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

    def set_eval_mode(self, eval_setting: bool):
        """Turns on/off any tuning"""
        raise NotImplementedError
