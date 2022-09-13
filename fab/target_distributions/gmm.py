from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn
import torch.nn.functional as f
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
    importance_weighted_expectation, effective_sample_size_over_p, setup_quadratic_function


class GMM(nn.Module, TargetDistribution):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0,
                 n_test_set_samples=1000, use_gpu=True,
                 true_expectation_estimation_n_samples=int(1e7)):
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
                                                             true_expectation_estimation_n_samples
                                                                              ))
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


def save_gmm_as_numpy(target: GMM):
    """Save params of GMM problem."""
    import pickle
    x_shift, A, b = setup_quadratic_function(torch.ones(target.dim), seed=0)
    params = {"mean": target.locs.numpy(),
              "scale_tril": target.scale_trils.numpy(),
              "true_expectation": target.true_expectation.numpy(),
              "expectation_x_shift": x_shift.numpy(),
              "expectation_A": A.numpy(),
              "expectation_b": b.numpy()
              }
    with open("gmm_problem.pkl", "wb") as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from fab.utils.plotting import plot_contours
    torch.manual_seed(0)
    loc_scaling = 40
    target = GMM(dim=2, n_mixes=40, loc_scaling=40.0)
    save_gmm_as_numpy(target) # Used for evaluating CRAFT
    plotting_bounds = (-loc_scaling * 1.4, loc_scaling* 1.4)
    plot_contours(target.log_prob, bounds=plotting_bounds, n_contour_levels=50,
                  grid_width_n_points=200)
    plt.show()
