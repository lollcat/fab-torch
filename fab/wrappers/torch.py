from typing import Tuple

import torch
from fab.types_ import Distribution


class WrappedTorchDist(Distribution):
    def __init__(self, torch_dist: torch.distributions.Distribution):
        self._torch_dist = torch_dist

    def sample_and_log_prob(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self._torch_dist.sample(shape)
        log_prob = self._torch_dist.log_prob(samples)
        return samples, log_prob

    def sample(self, shape: Tuple) -> torch.Tensor:
        return self._torch_dist.sample(shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._torch_dist.log_prob(x)

    def event_shape(self) -> Tuple[int, ...]:
        return self._torch_dist.event_shape
