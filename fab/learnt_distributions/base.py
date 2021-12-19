import torch.nn as nn
from fab.types import Distribution

class TrainableDistribution(Distribution, nn.Module):
    """Base class for trainable distributions."""
    pass
