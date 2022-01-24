import torch.nn as nn
from fab.types_ import Distribution

class TrainableDistribution(Distribution, nn.Module):
    """Base class for trainable distributions."""
