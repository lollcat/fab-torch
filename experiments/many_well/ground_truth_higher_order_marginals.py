import numpy as np
import matplotlib.pyplot as plt
import torch

from fab.target_distributions.many_well import ManyWellEnergy
from fab.utils.plotting import plot_marginal_pair


if __name__ == '__main__':
    torch.manual_seed(3)
    dist = ManyWellEnergy(dim=32)
    samples = dist.sample((int(2e4),))
    double_well_marginal = samples[:, np.arange(16)*2]
    plot_marginal_pair(double_well_marginal)
    plt.show()

    double_well_light_marginal_per_dim = (double_well_marginal < -1.) & (double_well_marginal > -2.1)

    # Plot rate of sampling fainter modes.
    n_light_mode = np.sum(double_well_light_marginal_per_dim.numpy(), axis=-1)
    plt.hist(n_light_mode, density=True, log=True, bins=np.arange(16+1))
    plt.xlabel("number of dim containing light mode (out of the 16 bimodal dim)")
    plt.ylabel("normalized frequency of samples")
    plt.title("Exact samples using rejection sampling")
    plt.show()


    # Plot higher order marginal.
    # Get marginal for the first two being light.
    marginal_condition = (double_well_light_marginal_per_dim[:, 0] == True) & (double_well_light_marginal_per_dim[:, 1] == True)
    marginal_samples = double_well_marginal[marginal_condition]
    n_rows = 4
    fig, axs = plt.subplots(n_rows, n_rows, sharex=True, sharey=True, figsize=(n_rows * 3, n_rows * 3))
    i_start = 2
    j_start = i_start + n_rows
    for i in range(n_rows):
        for j in range(n_rows):
            plot_marginal_pair(marginal_samples, ax=axs[i, j], marginal_dims=(i + i_start, j + j_start), bounds=(-3, 3),
                               alpha=0.2)
    plt.show()




