from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn
import torch.nn.functional as f
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
    importance_weighted_expectation
import numpy as np


class GMM(nn.Module, TargetDistribution):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0,
                 n_test_set_samples=500, use_gpu=True):
        super(GMM, self).__init__()
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        mean = (torch.rand((n_mixes, dim)) - 0.5)*2 * loc_scaling
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(f.softplus(log_var)))
        self.distribution = self.get_distribution
        self.expectation_function = quadratic_function
        self.true_expectation = MC_estimate_true_expectation(self,
                                                             self.expectation_function,
                                                             int(1e6)).item()
        self.device = "cuda" if use_gpu else "cpu"
        self.to(self.device)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
                self.distribution = self.get_distribution
        else:
            self.cpu()
            self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs,
                                                     scale_tril=self.scale_trils)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))

    def log_prob(self, x: torch.Tensor):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:
        expectation = importance_weighted_expectation(self.expectation_function,
                                                         samples, log_w)
        bias_normed = np.abs(expectation - self.true_expectation) / self.true_expectation
        if log_q_fn:
            test_mean_log_prob = torch.mean(log_q_fn(self.test_set))
            summary_dict = {"test_set_mean_log_prob": test_mean_log_prob.item(),
                            "bias_normed": bias_normed.item()}
        else:
            summary_dict = {"bias_normed": bias_normed.item()}
        return summary_dict


if __name__ == '__main__':
    target = GMM(dim=2, n_mixes=2, loc_scaling=1.0)
    target.test_set