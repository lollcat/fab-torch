import pytest

from fab.sampling_methods.transition_operators.metropolis import Metropolis
import pytest
import torch

from fab.types import LogProbFunc




def test_metroplis_estimate_easy_mean(dim: int = 2):
    """Test that Metrpolis can be used to estimate the mean to a reasonable degree of accuracy
    for an easy Guassian distribution."""
    # define base and target distribution
    loc_target = torch.ones(dim)
    loc_base = torch.ones(dim) - 1
    scale_tril = torch.eye(dim)
    target_log_prob = torch.distributions.MultivariateNormal(loc=loc_target,
                                                             scale_tril=scale_tril).log_prob
    base_dist = torch.distributions.MultivariateNormal(loc=loc_base,
                                                             scale_tril=scale_tril)
    # initialise metropolis transition operator
    metropolis_transition = Metropolis()



