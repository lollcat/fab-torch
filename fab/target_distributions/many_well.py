import torch
import torch.nn as nn
import itertools
import numpy as np

class Energy(torch.nn.Module):
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]


class DoubleWellEnergy(Energy, nn.Module):
    def __init__(self, dim, a=0.0, b=-4., c=1.):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2

    def log_prob(self, x):
        return torch.squeeze(-self.energy(x))

class ManyWellEnergy(DoubleWellEnergy):
    def __init__(self, dim=4, *args, **kwargs):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        super(ManyWellEnergy, self).__init__(dim=2, *args, **kwargs)
        self.dim = dim
        self.centre = 1.7
        self.max_dim_for_all_modes = 40 # otherwise we get memory issues on huuuuge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = torch.meshgrid([torch.tensor([-self.centre, self.centre])for _ in range(self.n_wells)])
            dim_1_vals = torch.stack([torch.flatten(dim) for dim in dim_1_vals_grid], dim=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            self.test_set__ = torch.zeros((n_modes, dim))
            self.test_set__[:, torch.arange(dim) % 2 == 0] = dim_1_vals
        else:
            print("using test set containing not all modes to prevent memory issues")

        self.true_energy_difference = 1.73 # calculated by by evaluating linspace of points changing x1, setting x2=0
        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]

    @property
    def test_set_(self):
        if self.dim < self.max_dim_for_all_modes:
            return self.test_set__
        else:
            batch_size = int(1e3)
            test_set = torch.zeros((batch_size, self.dim))
            test_set[:, torch.arange(self.dim) % 2 == 0] = \
                -self.centre + self.centre * 2 * torch.randint(high=2, size=(batch_size, int(self.dim/2)))
            return test_set

    def test_set(self, device):
        return (self.test_set_ + torch.randn_like(self.test_set_)*0.2).to(device)

    def log_prob(self, x):
        return torch.sum(
            torch.stack(
                [super(ManyWellEnergy, self).log_prob(x[:, i*2:i*2+2]) for i in range(self.n_wells)]),
            dim=0)

    def log_prob_2D(self, x):
        # for plotting, given 2D x
        return super(ManyWellEnergy, self).log_prob(x)

    @torch.no_grad()
    def performance_metrics(self, train_class, x_samples, log_w,
                            n_batches_stat_aggregation=20):
        return {}, {} # currently don't trust energy differences as useful
        assert x_samples.shape[0] % n_batches_stat_aggregation == 0
        samples_per_batch = x_samples.shape[0] // n_batches_stat_aggregation
        free_energy_differences = np.empty((n_batches_stat_aggregation, self.n_wells))
        for i, batch_number in enumerate(range(n_batches_stat_aggregation)):
            if i != n_batches_stat_aggregation - 1:
                log_w_batch = log_w[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
                x_samples_batch = x_samples[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch]
            else:
                log_w_batch = log_w[batch_number * samples_per_batch:]
                x_samples_batch = x_samples[batch_number * samples_per_batch:]
            for well_number in range(self.n_wells):
                dim1_index = well_number*2
                relevant_x_samples_indices_lower_well = (x_samples_batch[:, dim1_index] < self.shallow_well_bounds[1]) & \
                                                        (x_samples_batch[:, dim1_index] > self.shallow_well_bounds[0])
                relevant_x_samples_indices_deep_well = (x_samples_batch[:, dim1_index] < self.deep_well_bounds[1]) & \
                                                        (x_samples_batch[:, dim1_index] > self.deep_well_bounds[0])
                shallow_well_energy = - torch.logsumexp(log_w_batch[relevant_x_samples_indices_lower_well], dim=-1)
                deep_well_energy = - torch.logsumexp(log_w_batch[relevant_x_samples_indices_deep_well], dim=-1)
                free_energy_difference = (shallow_well_energy - deep_well_energy).cpu().numpy()
                free_energy_difference[np.isinf(free_energy_difference)] = np.nan
                free_energy_differences[batch_number, well_number] = free_energy_difference
        biases_normed = np.abs(np.nanmean(free_energy_differences, axis=0) - self.true_energy_difference)\
                 /self.true_energy_difference
        stds_normed = np.nanstd(free_energy_differences, axis=0)/self.true_energy_difference
        if self.n_wells > 1:
            summary_dict = {"mean_bias_normed": np.mean(biases_normed),
                            "max_bias_normed" : np.max(biases_normed),
                            "mean_std_normed" : np.mean(stds_normed),
                             "max_std_normed" : np.max(stds_normed)
                            }
        else:
            summary_dict = {"mean_bias_normed": np.mean(biases_normed),
                            "mean_std_normed": np.mean(stds_normed),
                            }
        long_dict = {"biases_normed": biases_normed, "stds_normed": stds_normed}
        return summary_dict, long_dict