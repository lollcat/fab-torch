import numpy as np
import matplotlib.pyplot as plt


from fab.target_distributions.many_well import ManyWellEnergy
from fab.utils.plotting import plot_marginal_pair


if __name__ == '__main__':
    dist = ManyWellEnergy(dim=32)
    samples = dist.sample((int(1e4),))
    double_well_marginal = samples[:, np.arange(16)*2]
    plot_marginal_pair(double_well_marginal)
    plt.show()

    double_well_marginal_bins = double_well_marginal > 0

    # Plot rate of sampling fainter modes.
    n_big_mode = np.sum(double_well_marginal_bins.numpy(), axis=-1)
    min_bin = 6
    plt.hist(n_big_mode, density=True, log=True, bins=np.arange(16 - min_bin + 1) + min_bin)
    plt.xlabel("number of dim containing heavier mode")
    plt.show()


    # Plot higher order marginal.
    # First a sanity check
    all_heavy_mode = n_big_mode == 16
    # plot_marginal_pair(double_well_marginal[all_heavy_mode])
    # plt.show()
    assert double_well_marginal[all_heavy_mode].min() > 0


    # Get marginal for the first two being light.
    marginal_condition = (double_well_marginal_bins[:, -1] == False) & (double_well_marginal_bins[:, -2] == False)
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




