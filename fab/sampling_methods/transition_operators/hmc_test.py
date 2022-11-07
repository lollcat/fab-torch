import torch

torch.autograd.set_detect_anomaly(True)
from fab.sampling_methods.transition_operators.hmc import HamiltonianMonteCarlo
from fab.sampling_methods.transition_operators.testing_utils import test_transition_operator, \
    TransitionOperatorTestConfig


def test_hmc(
        config: TransitionOperatorTestConfig = TransitionOperatorTestConfig(),
        n_iterations: int = 50,
        batch_size: int = 64):
    mass_init = torch.ones(config.dim)
    hmc = HamiltonianMonteCarlo(
        n_ais_intermediate_distributions=config.n_ais_intermediate_distributions,
        dim=config.dim,
        base_log_prob=config.learnt_sampler.log_prob,
        target_log_prob=config.target.log_prob,
        alpha=config.alpha,
        p_target=config.p_target,
        n_outer=10,
        epsilon=1.0, L=5,
        mass_init=mass_init)
    test_transition_operator(transition_operator=hmc,
                             config=config,
                             n_iterations=n_iterations,
                             n_samples=batch_size
                             )

if __name__ == '__main__':
    test_hmc()
