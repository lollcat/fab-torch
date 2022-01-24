from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
    importance_weighted_expectation
import numpy as np


class GMM(nn.Module, TargetDistribution):
    # mog with random mean and var
    def __init__(self, dim: int = 2, n_mixes: int = 5,
                 min_cov: float = 0.5, loc_scaling: float = 3.0, diagonal_covariance: bool = True,
                 cov_scaling: float = 1.0, uniform_component_probs: bool = False,
                 n_test_set_samples: int = 1000):
        super(GMM, self).__init__()
        self.dim = dim
        self.n_mixes = n_mixes
        self.n_test_set_samples = n_test_set_samples
        self.distributions = []
        locs = []
        scale_trils = []
        for i in range(n_mixes):
            loc = torch.randn(dim)*loc_scaling
            if diagonal_covariance:
                scale_tril = torch.diag(torch.rand(dim)*cov_scaling + min_cov)
            else:
                # https://stackoverflow.com/questions/58176501/how-do-you-generate-positive-definite-matrix-in-pytorch
                Sigma_k = torch.rand(dim, dim) * cov_scaling + min_cov
                Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
                Sigma_k.add_(torch.eye(dim))
                scale_tril = torch.tril(Sigma_k)
            locs.append(loc[None, :])
            scale_trils.append(scale_tril[None, :, :])

        locs = torch.cat(locs)
        scale_trils = torch.cat(scale_trils)
        if uniform_component_probs:
             self.register_buffer("cat_probs", torch.ones(n_mixes))
        else:
            self.register_buffer("cat_probs", torch.rand(n_mixes))
        self.register_buffer("locs", locs)
        self.register_buffer("scale_trils", scale_trils)
        self.distribution = self.get_distribution
        self.expectation_function = quadratic_function
        self.true_expectation = MC_estimate_true_expectation(self,
                                                             self.expectation_function,
                                                             int(1e6)).item()

    def to(self, device):
        super(GMM, self).to(device)
        self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, scale_tril=self.scale_trils, )
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com)

    def log_prob(self, x: torch.Tensor):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))


    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None) -> Dict:
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