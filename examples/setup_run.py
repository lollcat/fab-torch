from typing import Union, Callable, Optional, List
import os
import pathlib
import wandb
from omegaconf import DictConfig

from datetime import datetime


from fab import Trainer, BufferTrainer, PrioritisedBufferTrainer
from fab.target_distributions.base import TargetDistribution
from fab.utils.logging import PandasLogger, WandbLogger, Logger, ListLogger
from fab.utils.replay_buffer import ReplayBuffer
from fab.utils.plotting import plot_history
import matplotlib.pyplot as plt
import torch

from fab import FABModel, HamiltoneanMonteCarlo, Metropolis
from examples.make_flow import make_wrapped_normflowdist

from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer


Plotter = Callable[[FABModel], List[plt.Figure]]
SetupPlotterFn = Callable[[DictConfig, TargetDistribution,
                           Optional[Union[ReplayBuffer, PrioritisedReplayBuffer]]], Plotter]


def get_n_iterations(
        n_training_iter: Union[int, None],
        n_flow_forward_pass: Union[int, None],
        loss_type: str,
        n_transition_operator_inner_steps: int,
        n_intermediate_ais_dist: int,
        transition_operator_type: str,
        use_buffer: bool,
        min_buffer_length: Optional[int] = None) -> int:
    """
    Calculate the number of training iterations, based on the run config.
    We define one "training iteration" as
        - for training by KLD: 1 forward pass of the flow to estimate KLD
        - for training by FAB: 1 forward pass of the flow and AIS
        - for training by FAB with buffer: 1 forward pass of the flow & AIS followed by
            n buffer sampling update steps.

    Note: We aim here to do the theoretical number of forward passes required for each method
    during training for fair comparison. Due to inefficiencies in implementation this will not match
    the actual number of flow forward passes.
    """
    assert bool(n_training_iter) != bool(n_flow_forward_pass)

    if n_training_iter:
        return n_training_iter
    else:
        if loss_type[0:4] == "flow":
            n_iter = n_flow_forward_pass
        else:
            if transition_operator_type == "hmc":
                # Note this also requires differentiating the flow, which is fair as the
                # KLD forward pass also requires a differentiation of target and flow step.
                n_flow_eval_per_ais_forward = \
                    n_transition_operator_inner_steps*n_intermediate_ais_dist
            else:
                assert transition_operator_type == "metropolis"
                n_flow_eval_per_ais_forward = n_transition_operator_inner_steps*n_intermediate_ais_dist
            if use_buffer:
                buffer_init_flow_eval = n_flow_eval_per_ais_forward * min_buffer_length
                # we do another ais evaluation per iteration to calculate the log prob of the
                # samples from the buffer.
                n_flow_eval_per_iter = n_flow_eval_per_ais_forward + 1
            else:
                buffer_init_flow_eval = 0
                n_flow_eval_per_iter = n_flow_eval_per_ais_forward
            n_iter = int((n_flow_forward_pass - buffer_init_flow_eval) / n_flow_eval_per_iter)
    print(f"{n_iter} iter for {n_flow_forward_pass} flow forward passes")
    return n_iter






def setup_logger(cfg: DictConfig, save_path: str) -> Logger:
    if hasattr(cfg.logger, "pandas_logger"):
        logger = PandasLogger(save=True,
                              save_path=save_path + "logging_hist.csv",
                              save_period=cfg.logger.pandas_logger.save_period)
    elif hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    elif hasattr(cfg.logger, "list_logger"):
        logger = ListLogger(save_path=save_path + "logging_hist.pkl")
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger


def setup_buffer(cfg: DictConfig, fab_model: FABModel) -> Union[ReplayBuffer,
                                                                PrioritisedReplayBuffer]:
    dim = cfg.target.dim  # applies to flow and target
    if cfg.training.prioritised_buffer is False:
        def initial_sampler():
            # used to fill the replay buffer up to its minimum size
            x, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                cfg.training.batch_size, logging=False)
            return x, log_w

        buffer = ReplayBuffer(dim=dim, max_length=cfg.training.maximum_buffer_length,
                              min_sample_length=cfg.training.min_buffer_length,
                              initial_sampler=initial_sampler,
                              temperature=cfg.training.buffer_temp)
    else:
        # buffer
        def initial_sampler():
            x, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                cfg.training.batch_size, logging=False)
            log_q_x = fab_model.flow.log_prob(x).detach()
            return x, log_w, log_q_x

        buffer = PrioritisedReplayBuffer(dim=dim, max_length=cfg.training.maximum_buffer_length,
                                         min_sample_length=cfg.training.min_buffer_length,
                                         initial_sampler=initial_sampler)
    return buffer


def setup_trainer_and_run(cfg: DictConfig, setup_plotter: SetupPlotterFn,
                          target: TargetDistribution):
    """Create and trainer and run."""
    dim = cfg.target.dim  # applies to flow and target
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.training.seed)
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = cfg.evaluation.save_path + current_time + "/"
    logger = setup_logger(cfg, save_path)
    if hasattr(cfg.logger, "wandb"):
        # if using wandb then save to wandb path
        save_path = os.path.join(wandb.run.dir, save_path)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)


    with open(save_path + "config.txt", "w") as file:
        file.write(str(cfg))

    flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                     layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                     act_norm=cfg.flow.act_norm)


    if cfg.fab.transition_operator.type == "hmc":
        # very lightweight HMC.
        transition_operator = HamiltoneanMonteCarlo(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            n_outer=1,
            epsilon=1.0, L=cfg.fab.transition_operator.n_inner_steps, dim=dim,
            step_tuning_method="p_accept")
    elif cfg.fab.transition_operator.type == "metropolis":
        transition_operator = Metropolis(n_transitions=cfg.fab.n_intermediate_distributions,
                                         n_updates=cfg.fab.transition_operator.n_inner_steps,
                                         adjust_step_size=True)
    else:
        raise NotImplementedError


    # use GPU if available
    if torch.cuda.is_available() and cfg.training.use_gpu:
      flow.cuda()
      transition_operator.cuda()
      print("utilising GPU")


    fab_model = FABModel(flow=flow,
                         target_distribution=target,
                         n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                         transition_operator=transition_operator,
                         loss_type=cfg.fab.loss_type)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.training.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    scheduler = None


    # Create buffer if needed
    if cfg.training.use_buffer is True:
        buffer = setup_buffer(cfg, fab_model)
    else:
        buffer = None

    plot = setup_plotter(cfg, target, buffer)

    # Create trainer
    if cfg.training.use_buffer is False:
        trainer = Trainer(model=fab_model, optimizer=optimizer, logger=logger, plot=plot,
                          optim_schedular=scheduler, save_path=save_path,
                          max_gradient_norm=cfg.training.max_grad_norm
                          )
    elif cfg.training.prioritised_buffer is False:
        trainer = BufferTrainer(model=fab_model, optimizer=optimizer, logger=logger, plot=plot,
                          optim_schedular=scheduler, save_path=save_path,
                                buffer=buffer,
                                n_batches_buffer_sampling=cfg.training.n_batches_buffer_sampling,
                                clip_ais_weights_frac=cfg.training.log_w_clip_frac,
                                max_gradient_norm=cfg.training.max_grad_norm
                                )
    else:
        trainer = PrioritisedBufferTrainer(model=fab_model, optimizer=optimizer, logger=logger,
                                           plot=plot,
                          optim_schedular=scheduler, save_path=save_path,
                                buffer=buffer,
                                n_batches_buffer_sampling=cfg.training.n_batches_buffer_sampling,
                                max_gradient_norm=cfg.training.max_grad_norm,
                                w_adjust_max_clip=cfg.training.w_adjust_max_clip
                                )

    n_iterations = get_n_iterations(
                n_training_iter=cfg.training.n_iterations,
                n_flow_forward_pass=cfg.training.n_flow_forward_pass,
                loss_type=cfg.fab.loss_type,
                n_transition_operator_inner_steps=cfg.fab.transition_operator.n_inner_steps,
                n_intermediate_ais_dist=cfg.fab.n_intermediate_distributions,
                transition_operator_type=cfg.fab.transition_operator.type,
                use_buffer=cfg.training.use_buffer,
                min_buffer_length=cfg.training.min_buffer_length,
                )

    trainer.run(n_iterations=n_iterations, batch_size=cfg.training.batch_size,
                n_plot=cfg.evaluation.n_plots,
                n_eval=cfg.evaluation.n_eval, eval_batch_size=cfg.evaluation.eval_batch_size,
                save=True, n_checkpoints=cfg.evaluation.n_checkpoints)
    if hasattr(cfg.logger, "list_logger"):
        plot_history(trainer.logger.history)
        plt.show()
        print(trainer.logger.history['eval_ess_flow_p_target'][-10:])
        print(trainer.logger.history['eval_ess_ais_p_target'][-10:])
        print(trainer.logger.history['test_set_mean_log_prob_p_target'][-10:])
