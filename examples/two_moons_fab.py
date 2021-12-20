import normflow as nf
import matplotlib.pyplot as plt
import torch

from fab import FABModel, HamiltoneanMonteCarlo, Trainer, Metropolis
from fab.utils.logging import ListLogger
from fab.utils.plotting import plot_history, plot_contours, plot_marginal_pair

from examples.make_flow import make_wrapped_normflowdist, make_wrapped_nflows_dist

FLOW_LIBS = ["normflow", "nflows"]

def train_fab(
        dim: int = 2,
        n_intermediate_distributions: int = 3,
        batch_size: int = 256,
        n_iterations: int = 5000,
        n_plots: int = 10,
        lr: float = 1e-4,
        transition_operator_type: str = "hmc",  # "metropolis",  "hmc",
        seed: int = 0,
        n_flow_layers: int = 5,
        flow_lib: str = FLOW_LIBS[0],
) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    if flow_lib == "normflow":
        flow = make_wrapped_normflowdist(dim)
    elif flow_lib == "nflows":
        flow = make_wrapped_nflows_dist(dim, n_flow_layers=n_flow_layers)
    else:
        raise NotImplementedError
    target = nf.distributions.target.TwoMoons()
    # setup transition operator
    if transition_operator_type == "hmc":
        # very lightweight HMC.
        transition_operator = HamiltoneanMonteCarlo(
            n_ais_intermediate_distributions=n_intermediate_distributions,
            n_outer=1,
            epsilon=1.0, L=2, dim=dim,
            step_tuning_method="p_accept")
    elif transition_operator_type == "metropolis":
        transition_operator = Metropolis(n_transitions=n_intermediate_distributions,
                                         n_updates=5, adjust_step_size=True)
    else:
        raise NotImplementedError
    fab_model = FABModel(flow=flow,
                         target_distribution=target,
                         n_intermediate_distributions=n_intermediate_distributions,
                         transition_operator=transition_operator)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    scheduler = None
    logger = ListLogger()


    # plot target
    plot_contours(target.log_prob)
    plt.show()

    # set up plotting
    fig, axs = plt.subplots(n_plots, 2, figsize=(6, n_plots*3), sharex=True, sharey=True)
    # define which iterations we will plot the progress on
    plot_number_iterator = iter(range(n_plots))


    def plot(fab_model, n_samples = 300):
        plot_index = next(plot_number_iterator)
        # plot flow samples
        samples_flow = fab_model.flow.sample((n_samples,))
        plot_marginal_pair(samples_flow, ax=axs[plot_index, 0])


        # plot ais samples
        samples_ais = fab_model.annealed_importance_sampler.sample_and_log_weights(n_samples,
                                                                                   logging=False)[0]
        plot_marginal_pair(samples_ais, ax=axs[plot_index, 1])
        if plot_index == 0:
            axs[plot_index, 0].set_title("flow samples")
            axs[plot_index, 1].set_title("ais samples")
        fig.show()

    # Create trainer
    trainer = Trainer(model=fab_model, optimizer=optimizer, logger=logger, plot=plot,
                      optim_schedular=scheduler)
    trainer.run(n_iterations=n_iterations, batch_size=batch_size, n_plot=n_plots)

    plot_history(logger.history)
    plt.show()



if __name__ == '__main__':
    train_fab()
