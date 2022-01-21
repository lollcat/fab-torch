from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn

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
    """Many Well target distribution create by repeating the Double Well Boltzmann distribution."""
    def __init__(self, dim=4, use_gpu: bool = True, *args, **kwargs):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        super(ManyWellEnergy, self).__init__(dim=2, *args, **kwargs)
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
            self.register_buffer("_test_set", test_set)
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


    @property
    def test_set(self):
        """Test set created from points manually placed near each mode."""
        if self.dim < self.max_dim_for_all_modes:
            return self._test_set.to(self.device)
        else:
            batch_size = int(1e3)
            test_set = torch.zeros((batch_size, self.dim), device=self.device)
            test_set[:, torch.arange(self.dim) % 2 == 0] = \
                -self.centre + self.centre * 2 * torch.randint(high=2,
                                                               size=(batch_size, int(self.dim/2)),
                                                               device=self.device)
            return test_set


    def log_prob(self, x):
        return torch.sum(
            torch.stack(
                [super(ManyWellEnergy, self).log_prob(x[:, i*2:i*2+2])
                 for i in range(self.n_wells)]),
            dim=0)

    def log_prob_2D(self, x):
        # for plotting, given 2D x
        return super(ManyWellEnergy, self).log_prob(x)

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None) -> Dict:
        del samples
        del log_w
        info = {"test_set_mean_log_prob": torch.mean(log_q_fn(self.test_set)).item()}
        return info
