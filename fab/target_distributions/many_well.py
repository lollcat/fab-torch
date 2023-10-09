from typing import Optional, Dict

import numpy as np

from fab.types_ import LogProbFunc

import torch
from fab.target_distributions.base import TargetDistribution
from fab.utils.training import DatasetIterator
from fab.sampling_methods import AnnealedImportanceSampler, HamiltonianMonteCarlo
from fab.wrappers.torch import WrappedTorchDist
from fab.target_distributions.double_well import DoubleWellEnergy



class ManyWellEnergy(DoubleWellEnergy, TargetDistribution):
    """Many Well target distribution create by repeating the Double Well Boltzmann distribution."""
    def __init__(self, dim=4, use_gpu: bool = True,
                 normalised: bool = False,
                 a=-0.5, b=-6.0, c=1.):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        super(ManyWellEnergy, self).__init__(dim=2, a=a, b=b, c=c)
        self.dim = dim
        self.centre = 1.7
        self.max_dim_for_all_modes = 40  # otherwise we get memory issues on huuuuge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = torch.meshgrid([torch.tensor([-self.centre, self.centre])for _ in
                                              range(self.n_wells)])
            dim_1_vals = torch.stack([torch.flatten(dim) for dim in dim_1_vals_grid], dim=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = torch.zeros((n_modes, dim))
            test_set[:, torch.arange(dim) % 2 == 0] = dim_1_vals
            self.register_buffer("_test_set_modes", test_set)
        else:
            print("using test set containing not all modes to prevent memory issues")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]

        if use_gpu:
            if torch.cuda.is_available():
                self.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self.normalised = normalised

    @property
    def log_Z(self):
        return torch.tensor(self.log_Z_2D * self.n_wells)

    @property
    def Z(self):
        return torch.exp(self.log_Z)


    def sample(self, shape):
        """Sample by sampling each pair of dimensions from the double well problem
        using rejection sampling for the first dimension, and exact sampling for the second. """
        return torch.concat([super(ManyWellEnergy, self).sample(shape)
                             for _ in range(self.n_wells)],
                            dim=-1)

    def get_modes_test_set_iterator(self, batch_size: int):
        """Test set created from points manually placed near each mode."""
        if self.dim < self.max_dim_for_all_modes:
            test_set = self._test_set_modes
        else:
            outer_batch_size = int(1e4)
            test_set = torch.zeros((outer_batch_size, self.dim))
            test_set[:, torch.arange(self.dim) % 2 == 0] = \
                -self.centre + self.centre * 2 * \
                torch.randint(high=2, size=(outer_batch_size, int(self.dim/2)))
        return DatasetIterator(batch_size=batch_size, dataset=test_set,
                               device=self.device)

    def log_prob(self, x):
        log_prob = torch.sum(
            torch.stack(
                [super(ManyWellEnergy, self).log_prob(x[:, i*2:i*2+2])
                 for i in range(self.n_wells)]),
            dim=0)
        if self.normalised:
            return log_prob - self.log_Z
        else:
            return log_prob

    def log_prob_2D(self, x):
        # for plotting, given 2D x
        return super(ManyWellEnergy, self).log_prob(x)

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:

        del samples
        n_runs = 50
        n_vals_per_split = log_w.shape[0] // n_runs
        log_w = log_w[:n_vals_per_split*n_runs]
        log_w = torch.stack(log_w.split(n_runs), dim=-1)

        # Check accuracy in estimating normalisation constant.
        log_Z_estimate = torch.logsumexp(log_w, dim=-1) - np.log(log_w.shape[-1])
        relative_error = torch.exp(log_Z_estimate - self.log_Z) - 1
        MSE_Z_estimate = torch.mean(torch.abs(relative_error))

        abs_MSE_log_Z_estimate = torch.mean(torch.abs(log_Z_estimate - self.log_Z))

        info = {}
        info.update(relative_MSE_Z_estimate=MSE_Z_estimate.cpu().item())
        info.update(abs_MSE_log_Z_estimate=abs_MSE_log_Z_estimate.cpu().item())

        if log_q_fn is not None:
            # Used later for estimation of test set probabilities.
            assert batch_size is not None
            n_batches = max(log_w.shape[0] // batch_size, 1)

            sum_log_prob = 0.0
            sum_log_prob_exact = 0.0
            sum_kl_exact = 0.0
            test_set_iterator_modes = self.get_modes_test_set_iterator(batch_size=batch_size)

            for x in test_set_iterator_modes:
                # Mode test set.
                log_q_x_modes = torch.sum(log_q_fn(x)).detach().cpu()
                sum_log_prob += log_q_x_modes

            for _ in range(n_batches):
                # Samples from p test set.
                x_exact = self.sample((batch_size,))
                log_q_x_exact = log_q_fn(x_exact)
                sum_log_prob_exact += torch.sum(log_q_x_exact).detach().cpu()
                sum_kl_exact += torch.sum(self.log_prob(x_exact) - self.log_Z - log_q_x_exact).detach().cpu()

            eval_batch_size = batch_size * n_batches

            info.update(
                test_set_modes_mean_log_prob=(sum_log_prob / test_set_iterator_modes.test_set_n_points).cpu().item(),
                test_set_exact_mean_log_prob=(sum_log_prob_exact / eval_batch_size).cpu().item(),
                forward_kl=(sum_kl_exact / eval_batch_size).cpu().item(),
                eval_batch_size=eval_batch_size
            )
        return info

if __name__ == '__main__':
    from fab.utils.plotting import plot_contours, plot_marginal_pair
    import matplotlib.pyplot as plt

    dim = 8
    target = ManyWellEnergy(dim=dim)
    samples = target.sample((1000,))
    samples_modes = target._test_set_modes


    plotting_bounds = (-3, 3)
    n_rows = dim // 2
    fig, axs = plt.subplots(dim // 2, 2, sharex=True, sharey=True, figsize=(10, n_rows * 3))

    for i in range(n_rows):
        plot_contours(target.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 0])
        plot_contours(target.log_prob_2D, bounds=plotting_bounds, ax=axs[i, 1])

        # plot flow samples
        plot_marginal_pair(samples, ax=axs[i, 0], bounds=plotting_bounds,
                           marginal_dims=(i * 2, i * 2 + 1))
        plot_marginal_pair(samples_modes, ax=axs[i, 1], bounds=plotting_bounds,
                           marginal_dims=(i * 2, i * 2 + 1))
        axs[i, 0].set_xlabel(f"dim {i * 2}")
        axs[i, 0].set_ylabel(f"dim {i * 2 + 1}")

        plt.tight_layout()
    axs[0, 0].set_title("target samples test set")
    axs[0, 1].set_title("mode samples test set")
    plt.show()
