from typing import Dict

import abc
import torch

from fab.types import LogProbFunc

class TransitionOperator(abc.ABC):

    def get_logging_info(self) -> Dict:
        """Returns a dictionary of information for logging purposes."""
        raise NotImplementedError

    @abc.abstractmethod
    def transition(self, x: torch.Tensor, log_p_x: LogProbFunc, i: int) -> torch.Tensor:
        """Returns x generated from transition with log_q_x, as the invariant
        distribution."""
        raise NotImplementedError
