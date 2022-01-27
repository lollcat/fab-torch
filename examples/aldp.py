import matplotlib.pyplot as plt
import torch

from fab import FABModel, HamiltoneanMonteCarlo, Trainer, Metropolis
from fab.utils.logging import ListLogger
from fab.utils.plotting import plot_history, plot_contours, plot_marginal_pair
from fab.target_distributions import AldpBoltzmann
from examples.make_flow import make_wrapped_normflowdist

if __name__ == '__main__':
    dim: int = 60
    n_intermediate_distributions: int = 2
    layer_nodes_per_dim = 5
    batch_size: int = 64
    n_iterations: int = int(1e4)
    n_eval = 100
    eval_batch_size = batch_size * 10
    n_plots: int = 0 # number of plots shows throughout tranining
    lr: float = 1e-4
    transition_operator_type: str = "hmc"  # "metropolis" or "hmc"
    seed: int = 0
    n_flow_layers: int = 10
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    flow = make_wrapped_normflowdist(dim, n_flow_layers=n_flow_layers,
                                     layer_nodes_per_dim=layer_nodes_per_dim,
                                    act_norm=True)
    target = AldpBoltzmann()

    if transition_operator_type == "hmc":
        # very lightweight HMC.
        transition_operator = HamiltoneanMonteCarlo(
            n_ais_intermediate_distributions=n_intermediate_distributions,
            n_outer=1,
            epsilon=1.0, L=5, dim=dim,
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
    logger = ListLogger()  # save training history

    plot = lambda *args, **kwargs: None

    trainer = Trainer(model=fab_model, optimizer=optimizer, logger=logger, plot=plot,
                      optim_schedular=scheduler)

    trainer.run(n_iterations=n_iterations, batch_size=batch_size, n_plot=n_plots,
                n_eval=n_eval, eval_batch_size=eval_batch_size)

    plot_history(logger.history)
    plt.show()


