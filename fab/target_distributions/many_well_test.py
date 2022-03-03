from fab.target_distributions.many_well import ManyWellEnergy
import torch

def test_many_well(dim: int = 6):
    assert dim % 2 == 0
    target = ManyWellEnergy(dim, a=-0.5, b=-6)
    sampler = torch.distributions.MultivariateNormal(loc=torch.ones(dim),
                                                              scale_tril=torch.eye(dim))
    samples = sampler.sample((100,))
    log_probs = target.log_prob(samples)

    assert torch.isfinite(log_probs).all()

    target.performance_metrics(samples=samples, log_w=torch.ones(samples.shape[0]),
                               log_q_fn=sampler.log_prob, batch_size=10)