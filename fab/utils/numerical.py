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


def effective_sample_size(log_w: torch.Tensor, normalised=False):
    # effective sample size, see https://arxiv.org/abs/1602.03572
    assert len(log_w.shape) == 1
    if not normalised:
        log_w = F.softmax(log_w, dim=0)
    return 1 / torch.sum(log_w ** 2) / log_w.shape[0]

def effective_sample_size_over_p(log_w: torch.Tensor):
    """Estimate the effective sample size using samples from the target.
    Note that these log weights must be calculated with a normalised target log prob."""
    # effective sample size, see https://arxiv.org/abs/1602.03572
    assert len(log_w.shape) == 1
    return 1 / torch.mean(torch.exp(log_w))


def setup_quadratic_function(x: torch.Tensor, seed: int = 0):
    # Useful for porting this problem to non torch libraries.
    torch.random.manual_seed(seed)
    # example function that we may want to calculate expectations over
    x_shift = 2 * torch.randn(x.shape[-1])
    A = 2 * torch.rand((x.shape[-1], x.shape[-1])).to(x.device)
    b = torch.rand(x.shape[-1]).to(x.device)
    torch.seed()  # set back to random number
    if x.dtype == torch.float64:
        return x_shift.double(), A.double(), b.double()
    else:
        assert x.dtype == torch.float32
        return x_shift, A, b


def quadratic_function(x: torch.Tensor, seed: int = 0):
    x_shift, A, b = setup_quadratic_function(x, seed)
    x = x + x_shift
    return torch.einsum("bi,ij,bj->b", x, A, x) + torch.einsum("i,bi->b", b, x)



def importance_weighted_expectation(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                                    log_w: torch.Tensor) -> torch.Tensor:
    normalised_importance_weights = F.softmax(log_w, dim=-1)
    function_values = f(x)
    expectation = normalised_importance_weights.T @ function_values
    return expectation
