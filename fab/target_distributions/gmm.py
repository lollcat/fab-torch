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
        self.expectation_function = quadratic_function
        self.register_buffer("true_expectation", MC_estimate_true_expectation(self,
                                                             self.expectation_function,
                                                             int(1e6)))
        self.device = "cuda" if use_gpu else "cpu"
        self.to(self.device)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs,
                                                     scale_tril=self.scale_trils,
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))

    def log_prob(self, x: torch.Tensor):
        log_prob = self.distribution.log_prob(x)
        # Very low probability samples can cause issues (we turn off validate_args of the
        # distribution object which typically raises an expection related to this.
        # We manually decrease the distributions log prob to prevent them having an effect on
        # the loss/buffer.
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = - torch.tensor(float("inf"))
        log_prob = log_prob + mask
        return log_prob

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:
        expectation = importance_weighted_expectation(self.expectation_function,
                                                         samples, log_w)
        true_expectation = self.true_expectation.to(expectation.device)
        bias_normed = np.abs(expectation - true_expectation) / true_expectation
        if log_q_fn:
            test_mean_log_prob = torch.mean(log_q_fn(self.test_set))
            summary_dict = {"test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                            "bias_normed": bias_normed.cpu().item()}
        else:
            summary_dict = {"bias_normed": bias_normed.cpu().item()}
        return summary_dict


if __name__ == '__main__':
    target = GMM(dim=2, n_mixes=2, loc_scaling=1.0)
    target.test_set