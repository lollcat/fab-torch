import torch

torch.autograd.set_detect_anomaly(True)
from fab.target_distributions.MoG import MoG
from fab.sampling_methods.transition_operators.hmc import HamiltoneanMonteCarlo
from fab.utils.plotting import plot_history
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_hmc():
    n_samples = 1000
    n_distributions_pretend = 3 # TODO: fix here and in hmc class to be 1
    dim = 2
    torch.manual_seed(2)
    target = MoG(dim=dim, n_mixes=3, loc_scaling=4)
    learnt_sampler =  torch.distributions.MultivariateNormal(loc=torch.zeros(dim),
                                                                 scale_tril=4*torch.eye(dim))

    hmc = HamiltoneanMonteCarlo(n_distributions=n_distributions_pretend, n_outer=5,
                                epsilon=1.0, L=5, dim=dim,
              step_tuning_method="p_accept")
    # "Expected_target_prob", "No-U", "p_accept"
    n_iterations = 50
    history = {}
    history.update(dict([(key, []) for key in hmc.interesting_info()]))
    for i in tqdm(range(n_iterations)):
        for j in range(n_distributions_pretend - 2):
            sampler_samples = learnt_sampler.sample_n(n_samples)
            x_HMC = hmc.transition(sampler_samples, target.log_prob, j)
        transition_operator_info = hmc.interesting_info()
        for key in transition_operator_info:
            history[key].append(transition_operator_info[key])
        if i == 0 or i == n_iterations - 1 or i == int(n_iterations / 2):
            x_HMC = x_HMC.cpu().detach()
            plt.plot(x_HMC[:, 0], x_HMC[:, 1], "o", alpha=0.5)
            plt.title(f"HMC samples, iteration {i}")
            plt.show()
    plot_history(history)
    plt.show()

    true_samples = target.sample((n_samples,)).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    plt.title("true samples")
    plt.show()

    sampler_samples = learnt_sampler.sample_n(n_samples).cpu().detach()
    plt.plot(sampler_samples[:, 0], sampler_samples[:, 1], "o", alpha=0.5)
    plt.title("sampler samples")
    plt.show()