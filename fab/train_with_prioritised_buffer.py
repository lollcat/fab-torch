from typing import Callable, Any, Optional, List

import torch.optim.optimizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from time import time
import os

from fab.utils.logging import Logger, ListLogger
from fab.core import FABModel
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer


lr_scheduler = Any  # a learning rate schedular from torch.optim.lr_scheduler
Plotter = Callable[[FABModel], List[plt.Figure]]

class PrioritisedBufferTrainer:
    """A trainer for the FABModel for use with a prioritised replay buffer, and a different form
    of loss. In this training loop we target p^2/q instead of p."""
    def __init__(self,
                 model: FABModel,
                 optimizer: torch.optim.Optimizer,
                 buffer: PrioritisedReplayBuffer,
                 n_batches_buffer_sampling: int = 2,
                 optim_schedular: Optional[lr_scheduler] = None,
                 logger: Logger = ListLogger(),
                 plot: Optional[Plotter] = None,
                 max_gradient_norm: Optional[float] = 5.0,
                 w_adjust_max_clip: float = 10,
                 w_adjust_in_buffer_after_update: bool = False,
                 save_path: str = ""):
        self.model = model
        self.optimizer = optimizer
        self.optim_schedular = optim_schedular
        self.logger = logger
        self.plot = plot
        # if no gradient clipping set max_gradient_norm to inf
        self.max_gradient_norm = max_gradient_norm if max_gradient_norm else float("inf")
        self.save_dir = save_path
        self.plots_dir = os.path.join(self.save_dir, f"plots")
        self.checkpoints_dir = os.path.join(self.save_dir, f"model_checkpoints")
        self.buffer = buffer
        self.n_batches_buffer_sampling = n_batches_buffer_sampling
        self.flow_device = next(model.flow.parameters()).device
        self.max_adjust_w_clip = w_adjust_max_clip
        self.w_adjust_in_buffer_after_update = w_adjust_in_buffer_after_update

        # instead of the standard target prob, we target p^2/q
        def ais_target_log_prob(x):
            return 2*self.model.target_distribution.log_prob(x) - self.model.flow.log_prob(x)

        self.ais_target_log_prob = ais_target_log_prob  # save
        self.model.annealed_importance_sampler.target_log_prob = ais_target_log_prob

    def save_checkpoint(self, i):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}/")
        pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
        self.model.save(os.path.join(checkpoint_path, "model.pt"))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(checkpoint_path, 'optimizer.pt'))
        self.buffer.save(os.path.join(checkpoint_path, 'buffer.pt'))
        if self.optim_schedular:
            torch.save(self.optim_schedular.state_dict(),
                       os.path.join(self.checkpoints_dir, 'scheduler.pt'))

    def make_and_save_plots(self, i, save):
        figures = self.plot(self.model)
        for j, figure in enumerate(figures):
            if save:
                figure.savefig(os.path.join(self.plots_dir, f"{j}_iter_{i}.png"))
            else:
                plt.show()
            plt.close(figure)

    def perform_eval(self, i, eval_batch_size, batch_size):
        # set ais distribution to target for evaluation of ess
        self.model.annealed_importance_sampler.transition_operator.set_eval_mode(True)
        self.model.annealed_importance_sampler.target_log_prob = self.model.target_distribution.log_prob
        eval_info_true_target = self.model.get_eval_info(outer_batch_size=eval_batch_size,
                                                         inner_batch_size=batch_size)
        # set ais distribution back to p^2/q.
        self.model.annealed_importance_sampler.target_log_prob = self.ais_target_log_prob
        eval_info_practical_target = self.model.get_eval_info(outer_batch_size=eval_batch_size,
                                                              inner_batch_size=batch_size)
        self.model.annealed_importance_sampler.transition_operator.set_eval_mode(False)
        eval_info = {}
        eval_info.update({key + "_p_target": val for key, val in eval_info_true_target.items()})
        eval_info.update(
            {key + "_p2overq_target": val for key, val in eval_info_practical_target.items()})

        eval_info.update(step=i)
        self.logger.write(eval_info)


    def run(self,
            n_iterations: int,
            batch_size: int,
            eval_batch_size: Optional[int] = None,
            n_eval: Optional[int] = None,
            n_plot: Optional[int] = None,
            n_checkpoints: Optional[int] = None,
            save: bool = True,
            tlimit: Optional[float] = None,
            start_time: Optional[float] = None) -> None:
        if save:
            pathlib.Path(self.plots_dir).mkdir(exist_ok=True)
            pathlib.Path(self.checkpoints_dir).mkdir(exist_ok=True)
        if n_checkpoints:
            checkpoint_iter = list(np.linspace(0, n_iterations - 1, n_checkpoints, dtype="int"))
        if n_eval is not None:
            eval_iter = list(np.linspace(0, n_iterations - 1, n_eval, dtype="int"))
            assert eval_batch_size is not None
        if n_plot is not None:
            plot_iter = list(np.linspace(0, n_iterations - 1, n_plot, dtype="int"))
        if tlimit is not None:
            assert n_checkpoints is not None, "Time limited specified but not checkpoints are " \
                                          "being saved."
        if start_time is not None:
            start_time = time()

        pbar = tqdm(range(n_iterations))
        max_it_time = 0.0
        for i in pbar:
            it_start_time = time()
            self.optimizer.zero_grad()
            # collect samples and log weights with AIS and add to the buffer
            x_ais, log_w_ais = self.model.\
                annealed_importance_sampler.sample_and_log_weights(batch_size)
            x_ais = x_ais.detach()
            log_w_ais = log_w_ais.detach()
            log_q_x_ais = self.model.flow.log_prob(x_ais)
            self.buffer.add(x_ais.detach(), log_w_ais.detach(), log_q_x_ais.detach())

            # we log info from the step of the recently generated ais points.
            info = self.model.get_iter_info()

            # We now take self.n_batches_buffer_sampling gradient steps using
            # data from the replay buffer.
            mini_dataset = self.buffer.sample_n_batches(
                    batch_size=batch_size, n_batches=self.n_batches_buffer_sampling)
            for (x, log_w, log_q_old, indices) in mini_dataset:
                x, log_w, log_q_old, indices = x.to(self.flow_device), log_w.to(self.flow_device), \
                                               log_q_old.to(self.flow_device), indices.to(self.flow_device)
                self.optimizer.zero_grad()
                log_q_x = self.model.flow.log_prob(x)
                # adjustment to account for change to theta since sample was last added/adjusted
                log_w_adjust = log_q_old - log_q_x.detach()
                w_adjust_pre_clip = torch.exp(log_w_adjust)  # no grad
                if self.max_adjust_w_clip is not None:
                    w_adjust = torch.clip(w_adjust_pre_clip, max=self.max_adjust_w_clip)
                else:
                    w_adjust = w_adjust_pre_clip
                # manually calculate the new form of the loss
                loss = - torch.mean(w_adjust * log_q_x)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.max_gradient_norm)
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        print("nan grad norm in replay step")
                else:
                    print("nan loss in replay step")

                # Adjust log weights in the buffer.
                if not self.w_adjust_in_buffer_after_update:
                    with torch.no_grad():
                        self.buffer.adjust(log_w_adjust, log_q_x, indices)


            info.update(loss=loss.cpu().detach().item(),
                        step=i,
                        grad_norm=grad_norm.cpu().detach().item(),
                        sampled_log_w_std=torch.std(log_w).detach().cpu().item(),
                        sampled_log_w_mean=torch.mean(log_w).detach().cpu().item(),
                        w_adjust_mean=torch.mean(w_adjust_pre_clip).detach().cpu().item(),
                        w_adjust_min=torch.min(w_adjust_pre_clip).detach().cpu().item(),
                        w_adjust_max=torch.max(w_adjust_pre_clip).detach().cpu().item(),
                        log_q_x_mean=torch.mean(log_q_x).cpu().item()
                        )

            if self.w_adjust_in_buffer_after_update:
                with torch.no_grad():
                    for (x, log_w, log_q_old, indices) in mini_dataset:
                        """Adjust importance weights in the buffer for the points in the `mini_dataset` 
                        to account for the updated theta."""
                        x, log_w, log_q_old, indices = x.to(self.flow_device), log_w.to(
                            self.flow_device), \
                                                       log_q_old.to(self.flow_device), indices.to(
                            self.flow_device)
                        log_q_new = self.model.flow.log_prob(x)
                        log_adjust_insert = log_q_old - log_q_new
                        self.buffer.adjust(log_adjust_insert, log_q_new, indices)
                    info.update(log_w_adjust_insert_mean = torch.mean(log_adjust_insert).detach().cpu().item(),
                                log_q_mean = torch.mean(log_q_new).detach().cpu().item())

            self.logger.write(info)
            pbar.set_description(f"loss: {loss.cpu().detach().item()}, ess base: {info['ess_base']},"
                                 f"ess ais: {info['ess_ais']}")

            if n_eval is not None:
                if i in eval_iter:
                    self.perform_eval(i, eval_batch_size, batch_size)

            if n_plot is not None:
                if i in plot_iter:
                    self.make_and_save_plots(i, save)

            if n_checkpoints is not None:
                if i in checkpoint_iter:
                    self.save_checkpoint(i)


            max_it_time = max(max_it_time, time() - it_start_time)

            # End job if necessary
            if tlimit is not None:
                time_past = (time() - start_time) / 3600
                if (time_past + max_it_time/3600) > tlimit:
                    # self.perform_eval(i, eval_batch_size, batch_size)
                    # self.make_and_save_plots(i, save)
                    self.save_checkpoint(i)
                    self.logger.close()
                    print(f"\nEnding training at iteration {i}, after training for {time_past:.2f} "
                          f"hours as timelimit {tlimit:.2f} hours has been reached.\n")
                    return

        print(f"\n Run completed in {(time() - start_time)/3600:.2f} hours")
        if tlimit is None:
            print("Timelimit not set")
        else:
            print(f"Run finished before timelimit of {tlimit:.2f} hours was reached.")

        self.logger.close()
