from fab.sampling_methods.transition_operators.base import TransitionOperator
import torch
import torch.nn as nn

class HamiltoneanMonteCarlo(nn.Module, TransitionOperator):
    pass