
from typing import Tuple

import torch
from nflows import flows

from fab.trainable_distributions import TrainableDistribution

class WrappedNFlowsModel(TrainableDistribution):
    """Wraps the distribution from nflows library
    (https://github.com/bayesiains/nflows) to work in this fab library."""

    def __init__(self, normalising_flow: flows.Flow):
        super(WrappedNFlowsModel, self).__init__()
        self._nf_model = normalising_flow

    def sample_and_log_prob(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) == 1
        return self._nf_model.sample_and_log_prob(num_samples=shape[0])

    def sample(self, shape: Tuple) -> torch.Tensor:
        assert len(shape) == 1
        return self._nf_model.sample(num_samples=shape[0])

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._nf_model.log_prob(x)

    @property
    def event_shape(self) -> Tuple[int, ...]:

        return self._nf_model.sample(1).shape[1:]  # kill first dimension which is the batch.
