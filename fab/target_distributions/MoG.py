import torch
import torch.nn as nn
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
    importance_weighted_expectation
import numpy as np


class MoG(nn.Module, TargetDistribution):
    # mog with random mean and var
    def __init__(self, dim: int =2, n_mixes: int =5,
                 min_cov: float=0.5, loc_scaling: float=3.0, diagonal_covariance=True,
                 cov_scaling=1.0, uniform_component_probs = False):
        super(MoG, self).__init__()
        self.dim = dim
        self.n_mixes = n_mixes
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
        self.true_expectation = MC_estimate_true_expectation(self, self.expectation_function, int(1e6)).item()
        print(f"true expectation is {self.true_expectation}")

    def to(self, device):
        super(MoG, self).to(device)
        self.distribution = self.get_distribution

    @property
    def get_distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs, scale_tril=self.scale_trils)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com)

    def log_prob(self, x: torch.Tensor):
        return self.distribution.log_prob(x)

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    @torch.no_grad()
    def performance_metrics(self, x_samples, log_w,
                            n_batches_stat_aggregation=10):
        samples_per_batch = x_samples.shape[0] // n_batches_stat_aggregation
        expectations = []
        for i, batch_number in enumerate(range(n_batches_stat_aggregation)):
            if i != n_batches_stat_aggregation - 1:
                log_w_batch = log_w[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
                x_samples_batch = x_samples[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
            else:
                log_w_batch = log_w[batch_number * samples_per_batch:]
                x_samples_batch = x_samples[batch_number * samples_per_batch:]
            expectation = importance_weighted_expectation(self.expectation_function,
                                                             x_samples_batch, log_w_batch).item()
            expectations.append(expectation)
        bias_normed = np.abs(np.mean(expectations) - self.true_expectation) / self.true_expectation
        std_normed = np.std(expectations) / self.true_expectation
        summary_dict = {"bias_normed": bias_normed, "std_normed": std_normed}
        return summary_dict