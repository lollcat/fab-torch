import torch

from fab.types import LogProbFunc

class TransitionOperator(object):
    def transition(self, x: torch.Tensor, log_q_x: LogProbFunc, i: int) -> torch.Tensor:
        """Returns x generated from transition with log_q_x, as the invariant
        distribution."""
        return NotImplemented
