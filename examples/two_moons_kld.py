

import normflow as nf
import matplotlib.pyplot as plt
import torch

from fab import Model, Trainer
from fab.utils.logging import ListLogger
from fab.utils.plotting import plot_history, plot_contours, plot_marginal_pair
from examples.make_flow.make_realnvp_normflow import make_normflow_model



class KLDModel(Model):
    def __init__(self, nf_model: nf.NormalizingFlow):
        self.nf_model = nf_model

    def loss(self, batch_size: int) -> torch.Tensor:
        return self.nf_model.reverse_kld(batch_size)

    def get_iter_info(self) -> dict:
        return {}


def train_kld(
        dim: int = 2,
        batch_size: int = 256,
        n_iterations: int = 500,
        n_plots: int = 10,
        lr: float = 1e-4,
        seed: int = 0) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    target = nf.distributions.target.TwoMoons()
    nf_model = make_normflow_model(dim, target)
    kld_model = KLDModel(nf_model)
    optimizer = torch.optim.Adam(nf_model.parameters(), lr=lr)
    logger = ListLogger()

    # plot target
    plot_contours(target.log_prob)
    plt.show()

    # set up plotting
    fig, axs = plt.subplots(n_plots, figsize=(6, n_plots*3), sharex=True, sharey=True)
    # define which iterations we will plot the progress on
    plot_number_iterator = iter(range(n_plots))


    def plot(kld_model, n_samples = 300):
        plot_index = next(plot_number_iterator)
        # plot flow samples
        samples_flow, _ = kld_model.nf_model.sample(n_samples)
        plot_marginal_pair(samples_flow, ax=axs[plot_index])
        fig.show()

    # Create trainer
    trainer = Trainer(model=kld_model, optimizer=optimizer, logger=logger, plot=plot)
    trainer.run(n_iterations=n_iterations, batch_size=batch_size, n_plot=n_plots)

    plot_history(logger.history)
    plt.show()


if __name__ == '__main__':
    train_kld()
