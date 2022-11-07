from fab.sampling_methods.transition_operators import Metropolis
from fab.sampling_methods.transition_operators.testing_utils import test_transition_operator

import torch

torch.autograd.set_detect_anomaly(True)
from fab.sampling_methods.transition_operators.testing_utils import test_transition_operator, \
    TransitionOperatorTestConfig


def test_metropolis(
        config: TransitionOperatorTestConfig = TransitionOperatorTestConfig(),
        n_iterations: int = 50,
        batch_size: int = 64):
    """Test that Metrpolis can be used to estimate the mean to a reasonable degree of accuracy
    for an easy Guassian distribution."""
    # initialise metropolis transition operator
    metropolis_transition = Metropolis(
        n_ais_intermediate_distributions=config.n_ais_intermediate_distributions,
        dim=config.dim,
        base_log_prob=config.learnt_sampler.log_prob,
        target_log_prob=config.target.log_prob,
        p_target=config.p_target, alpha=config.alpha, n_updates=5)
    test_transition_operator(transition_operator=metropolis_transition,
                             config=config,
                             n_iterations=n_iterations,
                             n_samples=batch_size
                             )

if __name__ == '__main__':
    test_metropolis()
