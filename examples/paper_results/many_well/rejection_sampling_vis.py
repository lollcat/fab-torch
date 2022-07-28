# Copied from fab.sampling.rejection_sampling_test.
import torch
from fab.sampling_methods.rejection_sampling import rejection_sampling
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl

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
    mpl.rcParams['figure.dpi'] = 300
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('figure', titlesize=15)
    rc('axes', titlesize=13, labelsize=13)  # fontsize of the axes title and labels
    #rc('legend', fontsize=6)
    rc('xtick', labelsize=11)
    rc('ytick', labelsize=11)


    # First plot contours so make sure that our rejection sampling meets condition that kq > p.
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    x = torch.linspace(-4, 4, 200)
    p = torch.exp(target_log_prob(x))
    kq = k*torch.exp(proposal.log_prob(x))
    axs[0].plot(x, p, label="p")
    axs[0].plot(x, kq, label="kq")
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_ylabel(r"$f(x_1)$")
    axs[0].legend()

    n_samples = 10000
    samples = rejection_sampling(n_samples, proposal, target_log_prob, k)
    axs[1].plot(x, p/TARGET_Z, label="p (normalised)")
    axs[1].hist(samples, density=True, bins=100, label="sample density")
    axs[1].legend()
    axs[1].set_xlabel(r"$x_1$")
    axs[1].set_ylabel("PDF")
    plt.tight_layout()

    fig.savefig("plots/rejection_sampling.png", bbox_inches="tight")
    plt.show()