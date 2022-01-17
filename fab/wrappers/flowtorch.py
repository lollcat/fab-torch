
from typing import Tuple

import torch

from fab.trainable_distributions import TrainableDistribution

class WrappedFlowTorchModel(TrainableDistribution):
    # TODO: write
    """Wraps the distribution from flowtorch library
    (https://github.com/facebookincubator/flowtorch) to work in this fab library."""

    def __init__(self, normalising_flow: NormalizingFlow):
        super(WrappedFlowTorchModel, self).__init__()
        self._nf_model = normalising_flow

    def sample_and_log_prob(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) == 1
        return self._nf_model.sample(shape[0])

    def sample(self, shape: Tuple) -> torch.Tensor:
        return self.sample_and_log_prob(shape)[0]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._nf_model.log_prob(x)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._nf_model.q0.shape
