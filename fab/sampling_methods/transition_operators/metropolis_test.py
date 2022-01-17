from fab.sampling_methods.transition_operators import Metropolis
from fab.sampling_methods.transition_operators.testing_utils import test_transition_operator


def test_metropolis_estimate_easy_mean(
        dim: int = 2,
        n_ais_intermediate_distributions: int = 2,
        n_iterations: int = 30) -> None:
    """Test that Metrpolis can be used to estimate the mean to a reasonable degree of accuracy
    for an easy Guassian distribution."""

    # initialise metropolis transition operator
    metropolis_transition = Metropolis(n_transitions=n_ais_intermediate_distributions, n_updates=5)
    test_transition_operator(metropolis_transition,
                             n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                             dim=dim, n_iterations=n_iterations)



