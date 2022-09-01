from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn
import torch.nn.functional as f
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
    importance_weighted_expectation, effective_sample_size_over_p


class Gaussian(nn.Module, TargetDistribution):
    def __init__(self, mean: torch.Tensor, scale: Optional[torch.Tensor] = None, seed=0,
                 n_test_set_samples: int = 1000, use_gpu: bool = True,
                 true_expectation_estimation_n_samples=int(1e7)):
        super(Gaussian, self).__init__()
        assert len(mean.shape) == 1
        if scale is not None:
            assert len(scale.shape) in [1, 2]
        self.seed = seed
        self.dim = mean.shape[0]
        self.n_test_set_samples = n_test_set_samples
        self.register_buffer("locs", mean)
        self.register_buffer("scale_tril", torch.diag(scale if scale else torch.ones_like(mean)))
        self.expectation_function = quadratic_function
        self.register_buffer("true_expectation", MC_estimate_true_expectation(self,
                                                             self.expectation_function,
                                                             true_expectation_estimation_n_samples
                                                                              ))
        self.device = "cuda" if use_gpu else "cpu"
        self.to(self.device)

    @property
    def distribution(self):
        dist = torch.distributions.MultivariateNormal(self.locs,
                                                      scale_tril=self.scale_tril,
                                                      validate_args=False)
        return dist

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()


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

    def evaluate_expectation(self, samples, log_w):
        expectation = importance_weighted_expectation(self.expectation_function,
                                                         samples, log_w)
        true_expectation = self.true_expectation.to(expectation.device)
        bias_normed = (expectation - true_expectation) / true_expectation
        return bias_normed

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:
        bias_normed = self.evaluate_expectation(samples, log_w)
        bias_no_correction = self.evaluate_expectation(samples, torch.ones_like(log_w))
        if log_q_fn:
            log_q_test = log_q_fn(self.test_set)
            log_p_test = self.log_prob(self.test_set)
            test_mean_log_prob = torch.mean(log_q_test)
            kl_forward = torch.mean(log_p_test - log_q_test)
            ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
            summary_dict = {
                "test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                "bias_normed": torch.abs(bias_normed).cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
                "ess_over_p": ess_over_p.detach().cpu().item(),
                "kl_forward": kl_forward.detach().cpu().item()
                            }
        else:
            summary_dict = {"bias_normed": bias_normed.cpu().item(),
                            "bias_no_correction": torch.abs(bias_no_correction).cpu().item()}
        return summary_dict

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from fab.utils.plotting import plot_contours
    torch.manual_seed(0)
    mean = torch.ones(2)
    target = Gaussian(mean=mean)
    plotting_bounds = (-2, 2)
    plot_contours(target.log_prob, bounds=plotting_bounds, n_contour_levels=50,
                  grid_width_n_points=200)
    plt.show()

