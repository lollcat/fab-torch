from typing import Tuple

import torch
import torch.nn as nn

class LearntDistribubtion(nn.Module):

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def sample_and_log_prob(self, shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def sample(self, shape: Tuple) -> torch.Tensor:
        """Returns samples from the model."""
        raise NotImplemented
