import numpy as np
import torch
import torch.nn as nn
from fab.sampling_methods.rejection_sampling import rejection_sampling


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
    def __init__(self, dim=2, a=-0.5, b=-6.0, c=1.):
        assert dim == 2  # We only use the 2D version
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define proposal params
            self.register_buffer("component_mix", torch.tensor([0.2, 0.8]))
            self.register_buffer("means", torch.tensor([-1.7, 1.7]))
            self.register_buffer("scales", torch.tensor([0.5, 0.5]))

    def _energy_dim_1(self, x_1):
        return self._a * x_1 + self._b * x_1.pow(2) + self._c * x_1.pow(4)

    def _energy_dim_2(self, x_2):
        return 0.5 * x_2.pow(2)

    def _energy(self, x):
        x_1 = x[:, 0]
        x_2 = x[:, 1]
        e1 = self._energy_dim_1(x_1)
        e2 = self._energy_dim_2(x_2)
        return e1 + e2

    def log_prob(self, x):
        return torch.squeeze(-self.energy(x))

    def sample_first_dimension(self, shape):
        assert len(shape) == 1
        # see fab.sampling_methods.rejection_sampling_test.py
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define target.
            def target_log_prob(x):
                return -x ** 4 + 6 * x ** 2 + 1 / 2 * x

            TARGET_Z = 11784.50927

            # Define proposal
            mix = torch.distributions.Categorical(self.component_mix)
            com = torch.distributions.Normal(self.means, self.scales)

            proposal = torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                             component_distribution=com)

            k = TARGET_Z * 3

            samples = rejection_sampling(shape[0], proposal, target_log_prob, k)
            return samples
        else:
            raise NotImplementedError


    def sample(self, shape):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            dim1_samples = self.sample_first_dimension(shape)
            dim2_samples = torch.distributions.Normal(
                torch.tensor(0.0).to(dim1_samples.device),
                torch.tensor(1.0).to(dim1_samples.device)
            ).sample(shape)
            return torch.stack([dim1_samples, dim2_samples], dim=-1)
        else:
            raise NotImplementedError

    @property
    def log_Z_2D(self):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            log_Z_dim0 = np.log(11784.50927)
            log_Z_dim1 = 0.5 * np.log(2 * torch.pi)
            return log_Z_dim0 + log_Z_dim1
        else:
            raise NotImplementedError

if __name__ == '__main__':
    # Test that rejection sampling is work as desired.
    import matplotlib.pyplot as plt
    target = DoubleWellEnergy(2)


    x_linspace = torch.linspace(-4, 4, 200)

    Z_dim_1 = 11784.50927
    samples = target.sample((10000,))
    p_1 = torch.exp(-target._energy_dim_1(x_linspace))
    # plot first dimension vs normalised log prob
    plt.plot(x_linspace, p_1/Z_dim_1, label="p_1 normalised")
    plt.hist(samples[:, 0], density=True, bins=100, label="sample density")
    plt.legend()
    plt.show()

    # Now dimension 2.
    Z_dim_2 = (2 * torch.pi)**0.5
    p_2 = torch.exp(-target._energy_dim_2(x_linspace))
    # plot first dimension vs normalised log prob
    plt.plot(x_linspace, p_2/Z_dim_2, label="p_2 normalised")
    plt.hist(samples[:, 1], density=True, bins=100, label="sample density")
    plt.legend()
    plt.show()



