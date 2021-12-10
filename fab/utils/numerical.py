from typing import Union, Callable, Any

import torch
import torch.nn.functional as F

fab_distribution = Any  # TODO: define generic distribution type

def MC_estimate_true_expectation(distribution: Union[torch.distributions.Distribution,
                                                     fab_distribution],
                                 expectation_function: Callable,
                                 n_samples: int):
    # requires the distribution to be able to be sampled from
    x_samples = distribution.sample((n_samples,))
    f_x = expectation_function(x_samples)
    return torch.mean(f_x)


def quadratic_function(x: torch.Tensor, seed: int =0):
    # example function that we may want to calculate expectations over
    torch.manual_seed(seed)
    x_shift = 2*torch.randn(x.shape[-1]).to(x.device)
    A = 2*torch.rand((x.shape[-1], x.shape[-1])).to(x.device)
    b = torch.rand(x.shape[-1]).to(x.device)
    x = x + x_shift
    return torch.einsum("bi,ij,bj->b", x, A, x) + torch.einsum("i,bi->b", b, x)



def importance_weighted_expectation(log_w, x, f):
    normalised_importance_weights = F.softmax(log_w, dim=-1)
    function_values = f(x)
    expectation = normalised_importance_weights.T @ function_values
    return expectation


