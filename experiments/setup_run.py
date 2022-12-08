import re
import time
from typing import Union, Callable, Optional, List
import os
import pathlib
import wandb
import numpy as np
from omegaconf import DictConfig

from datetime import datetime

import matplotlib.pyplot as plt
import torch

from fab import Trainer, BufferTrainer, PrioritisedBufferTrainer
from fab.target_distributions.base import TargetDistribution
from fab.utils.logging import PandasLogger, WandbLogger, Logger, ListLogger
from fab.utils.replay_buffer import ReplayBuffer
from fab.utils.plotting import plot_history

from fab import FABModel, HamiltonianMonteCarlo, Metropolis
from fab.core import ALPHA_DIV_TARGET_LOSSES
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer

from experiments.make_flow import make_wrapped_normflow_realnvp, \
    make_wrapped_normflow_resampled_flow, make_wrapped_normflow_snf_model

Plotter = Callable[[FABModel], List[plt.Figure]]
SetupPlotterFn = Callable[[DictConfig, TargetDistribution,
                           Optional[Union[ReplayBuffer, PrioritisedReplayBuffer]]], Plotter]


def get_n_iterations(
        n_training_iter: Union[int, None],
        n_flow_forward_pass: Union[int, None],
        batch_size: int,
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
    # must specify either number of training iterations or flow forward passes.
    assert bool(n_training_iter) != bool(n_flow_forward_pass)

    if n_training_iter:
        return n_training_iter
    else:
        if loss_type[0:4] == "flow" or loss_type[:6] == "target":
            n_iter = n_flow_forward_pass // batch_size
        else:
            if transition_operator_type == "hmc":
                # Note this also requires differentiating the flow, which is fair as the
                # KLD forward pass also requires a differentiation of target and flow step.
                # +1 is for the initial sampling step.
                n_flow_eval_per_ais_forward = \
                    (n_transition_operator_inner_steps)*n_intermediate_ais_dist + 1
            else:
                assert transition_operator_type == "metropolis"
                # +1 for the initial sampling step
                n_flow_eval_per_ais_forward = \
                    n_transition_operator_inner_steps*n_intermediate_ais_dist + 1
            if use_buffer:
                buffer_init_flow_eval = n_flow_eval_per_ais_forward * min_buffer_length
                # we do another ais evaluation per iteration to calculate the log prob of the
                # samples from the buffer.
                n_flow_eval_per_iter = (n_flow_eval_per_ais_forward + 1)*batch_size
            else:
                buffer_init_flow_eval = 0
                n_flow_eval_per_iter = n_flow_eval_per_ais_forward*batch_size
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


def setup_buffer(cfg: DictConfig, fab_model: FABModel, auto_fill_buffer: bool) -> \
        Union[ReplayBuffer, PrioritisedReplayBuffer]:
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
            point, log_w = fab_model.annealed_importance_sampler.sample_and_log_weights(
                cfg.training.batch_size, logging=False)
            return point.x.detach(), log_w, point.log_q.detach()

        buffer = PrioritisedReplayBuffer(dim=dim, max_length=cfg.training.maximum_buffer_length,
                                         min_sample_length=cfg.training.min_buffer_length,
                                         initial_sampler=initial_sampler,
                                         fill_buffer_during_init=auto_fill_buffer)
    return buffer

def get_load_checkpoint_dir(outer_checkpoint_dir):
    """Get directory of most recent checkpoint"""
    try:
        # load the most recent checkpoint, from the most recent run.
        chkpts = [it.path for it in os.scandir(outer_checkpoint_dir) if it.is_dir()]
        folder_names = [it.name for it in os.scandir(outer_checkpoint_dir) if
                        it.is_dir()]
        times = [datetime.fromisoformat(time).timestamp() for time in folder_names]
        # grab most recent dir with argmax on times
        latest_chkpts_dir = os.path.join(chkpts[np.argmax(times)], "model_checkpoints")
        iter_dirs = [it.path for it in os.scandir(latest_chkpts_dir) if it.is_dir()]
        re_matches = [re.search(r"(.*iter_([0-9]*))", subdir) for subdir in iter_dirs]
        iter_numbers = [int(match.groups()[1]) if match else -1 for match in re_matches]
        chkpt_dir = re_matches[np.argmax(iter_numbers)].groups()[0]
        iter_number = np.max(iter_numbers)
    except:
        print("Starting training from the beginning with no checkpoint.")
        return None, 0
    return chkpt_dir, iter_number


def setup_model(cfg: DictConfig, target: TargetDistribution) -> FABModel:
    dim = cfg.target.dim  # applies to flow and target
    p_target = cfg.fab.loss_type not in ALPHA_DIV_TARGET_LOSSES or \
                         not cfg.training.prioritised_buffer
    if cfg.flow.resampled_base:
        flow = make_wrapped_normflow_resampled_flow(
            dim,
            n_flow_layers=cfg.flow.n_layers,
            layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
            act_norm=cfg.flow.act_norm)

    elif cfg.flow.use_snf:
        flow = make_wrapped_normflow_snf_model(
            dim,
            n_flow_layers=cfg.flow.n_layers,
            layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
            act_norm=cfg.flow.act_norm,
            target=target,
            mh_prop_scale=cfg.flow.snf.step_size,
            it_snf_layer=cfg.flow.snf.it_snf_layer,
            mh_steps=cfg.flow.snf.num_steps,
            transition_operator_type=cfg.flow.snf.transition_operator_type,
            hmc_n_leapfrog_steps=cfg.flow.snf.num_steps
                                      )
    else:
        flow = make_wrapped_normflow_realnvp(dim, n_flow_layers=cfg.flow.n_layers,
                                             layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                             act_norm=cfg.flow.act_norm)


    if cfg.fab.transition_operator.type == "hmc":
        # very lightweight HMC.
        transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            dim=dim,
            base_log_prob=flow.log_prob,
            target_log_prob=target.log_prob,
            alpha=cfg.fab.alpha,
            p_target=p_target,
            n_outer=1,
            epsilon=cfg.fab.transition_operator.init_step_size,
            L=cfg.fab.transition_operator.n_inner_steps,
            )

    elif cfg.fab.transition_operator.type == "metropolis":
        transition_operator = Metropolis(
            n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
            dim=dim,
            base_log_prob=flow.log_prob,
            target_log_prob=target.log_prob,
            p_target=p_target,
            alpha=cfg.fab.alpha,
            n_updates=cfg.fab.transition_operator.n_inner_steps,
            adjust_step_size=cfg.fab.transition_operator.tune_step_size,
            target_p_accept=cfg.fab.transition_operator.target_p_accept,
            min_step_size=cfg.fab.transition_operator.init_step_size,
            max_step_size=cfg.fab.transition_operator.init_step_size
        )
    else:
        raise NotImplementedError


    # use GPU if available
    if torch.cuda.is_available() and cfg.training.use_gpu:
        flow.cuda()
        transition_operator.cuda()
        print("\n*************  Utilising GPU  ****************** \n")
    else:
        print("\n*************  Utilising CPU  ****************** \n")


    fab_model = FABModel(flow=flow,
                         target_distribution=target,
                         n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                         transition_operator=transition_operator,
                         alpha=cfg.fab.alpha,
                         loss_type=cfg.fab.loss_type)
    return fab_model



def setup_trainer_and_run_flow(cfg: DictConfig, setup_plotter: SetupPlotterFn,
                          target: TargetDistribution):
    """Setup model and train."""
    if cfg.training.tlimit:
        start_time = time.time()
    else:
        start_time = None
    if cfg.training.checkpoint_load_dir is not None:
        if not os.path.exists(cfg.training.checkpoint_load_dir):
            print("no checkpoint loaded, starting training from scratch")
            chkpt_dir = None
            iter_number = 0
        else:
            chkpt_dir, iter_number = get_load_checkpoint_dir(cfg.training.checkpoint_load_dir)
    else:
        chkpt_dir = None
        iter_number = 0

    save_path = os.path.join(cfg.evaluation.save_path, str(datetime.now().isoformat()))
    logger = setup_logger(cfg, save_path)
    if hasattr(cfg.logger, "wandb"):
        # if using wandb then save to wandb path
        save_path = os.path.join(wandb.run.dir, save_path)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    n_iterations = get_n_iterations(
        n_training_iter=cfg.training.n_iterations,
        n_flow_forward_pass=cfg.training.n_flow_forward_pass,
        batch_size=cfg.training.batch_size,
        loss_type=cfg.fab.loss_type,
        n_transition_operator_inner_steps=cfg.fab.transition_operator.n_inner_steps,
        n_intermediate_ais_dist=cfg.fab.n_intermediate_distributions,
        transition_operator_type=cfg.fab.transition_operator.type,
        use_buffer=cfg.training.use_buffer,
        min_buffer_length=cfg.training.min_buffer_length,
    )
    print(f"running for {n_iterations}")
    cfg.training.n_iterations = n_iterations


    with open(os.path.join(save_path, "config.txt"), "w") as file:
        file.write(str(cfg))

    fab_model = setup_model(cfg, target)
    optimizer = torch.optim.Adam(fab_model.flow.parameters(), lr=cfg.training.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    scheduler = None

    # Create buffer if needed
    if cfg.training.use_buffer is True:
        buffer = setup_buffer(cfg, fab_model, auto_fill_buffer=chkpt_dir is None)
    else:
        buffer = None
    if chkpt_dir is not None:
        map_location = "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
        fab_model.load(os.path.join(chkpt_dir, "model.pt"), map_location)
        opt_state = torch.load(os.path.join(chkpt_dir, 'optimizer.pt'), map_location)
        optimizer.load_state_dict(opt_state)
        if buffer is not None:
            buffer.load(path=os.path.join(chkpt_dir, 'buffer.pt'))
            assert buffer.can_sample, "if a buffer is loaded, it is expected to contain " \
                                      "enough samples to sample from"
        print(f"\n\n****************loaded checkpoint: {chkpt_dir}*******************\n\n")

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
        trainer = PrioritisedBufferTrainer(
            model=fab_model,
            optimizer=optimizer,
            logger=logger,
            plot=plot,
            optim_schedular=scheduler,
            save_path=save_path,
            buffer=buffer,
            n_batches_buffer_sampling=cfg.training.n_batches_buffer_sampling,
            max_gradient_norm=cfg.training.max_grad_norm,
            w_adjust_max_clip=cfg.training.w_adjust_max_clip,
            alpha=cfg.fab.alpha
            )


    trainer.run(n_iterations=n_iterations,
                batch_size=cfg.training.batch_size,
                n_plot=cfg.evaluation.n_plots,
                n_eval=cfg.evaluation.n_eval,
                eval_batch_size=cfg.evaluation.eval_batch_size,
                save=True,
                n_checkpoints=cfg.evaluation.n_checkpoints,
                tlimit=cfg.training.tlimit,
                start_time=start_time,
                start_iter=iter_number)

    if hasattr(cfg.logger, "list_logger"):
        plot_history(trainer.logger.history)
        plt.show()
        print(trainer.logger.history['eval_ess_flow_p_target'][-10:])
        print(trainer.logger.history['eval_ess_ais_p_target'][-10:])
        print(trainer.logger.history['test_set_mean_log_prob_p_target'][-10:])

