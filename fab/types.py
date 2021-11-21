from typing import Callable
import torch

LogProbFunc = Callable[[torch.Tensor], torch.Tensor]
LogWeights = torch.Tensor
