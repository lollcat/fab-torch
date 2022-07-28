from typing import Callable

import torch


def rejection_sampling(n_samples: int, proposal: torch.distributions.Distribution,
                       target_log_prob_fn: Callable, k: float) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample((n_samples*10,))
    u_0 = torch.distributions.Uniform(0, k*torch.exp(proposal.log_prob(z_0)))\
        .sample().to(z_0)
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_log_prob_fn, k)
        samples = torch.concat([samples, new_samples], dim=0)
        return samples
