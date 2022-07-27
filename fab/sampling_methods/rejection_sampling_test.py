from typing import Callable
import torch
from fab.sampling_methods.rejection_sampling import rejection_sampling
import matplotlib.pyplot as plt

# Define target.
def target_log_prob(x):
    return -x**4 + 6 * x**2 + 1/2 * x

TARGET_Z = 11784.50927

# Define proposal
mix = torch.distributions.Categorical(torch.tensor([0.2, 0.8]))
com = torch.distributions.Normal(torch.tensor([-1.7, 1.7]), torch.tensor([0.5, 0.5]))

proposal = torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                             component_distribution=com)

k = TARGET_Z * 3




if __name__ == '__main__':
    # First plot contours so make sure that our rejection sampling meets condition that kq > p.
    x = torch.linspace(-4, 4, 200)
    p = torch.exp(target_log_prob(x))
    kq = k*torch.exp(proposal.log_prob(x))
    plt.plot(x, p, label="p")
    plt.plot(x, kq, label="kq")
    plt.legend()
    plt.show()
    assert (kq > p).all()

    n_samples = 10000
    samples = rejection_sampling(n_samples, proposal, target_log_prob, k)
    plt.plot(x, p/TARGET_Z, label="p normalised")
    plt.hist(samples, density=True, bins=100, label="sample density")
    plt.legend()
    plt.show()
    pass





