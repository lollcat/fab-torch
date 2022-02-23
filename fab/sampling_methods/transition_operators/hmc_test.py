import torch

torch.autograd.set_detect_anomaly(True)
from fab.sampling_methods.transition_operators.hmc import HamiltoneanMonteCarlo, \
    HMC_STEP_TUNING_METHODS
from fab.sampling_methods.transition_operators.testing_utils import test_transition_operator

# HMC_STEP_TUNING_METHODS = ["p_accept", "Expected_target_prob", "No-U", "No-U-unscaled"]

def test_hmc_(
        step_tuning_method: str = HMC_STEP_TUNING_METHODS[2],
        dim: int = 2,
        n_ais_intermediate_distributions: int = 2,
        n_iterations: int = 100,
        batch_size: int = 64):
    mass_init = torch.ones(dim)
    hmc = HamiltoneanMonteCarlo(n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                                n_outer=2,
                                epsilon=1.0, L=5, dim=dim,
                                step_tuning_method=step_tuning_method,
                                mass_init=mass_init)
    test_transition_operator(transition_operator=hmc,
                             n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                             dim=dim, n_iterations=n_iterations, n_samples=batch_size)
