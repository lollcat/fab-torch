from typing import Callable, Tuple
import torch
import abc

LogProbFunc = Callable[[torch.Tensor], torch.Tensor]


class Distribution(abc.ABC):
    """Used for distributions that have a defined sampling and log probability function."""
    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    @abc.abstractmethod
    def sample_and_log_prob(self, shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    @abc.abstractmethod
    def sample(self, shape: Tuple) -> torch.Tensor:
        """Returns samples from the model."""
        raise NotImplemented
