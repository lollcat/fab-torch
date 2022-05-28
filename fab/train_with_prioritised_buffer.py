from typing import Callable, Any, Optional, List

import torch.optim.optimizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pathlib
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

        # instead of the standard target prob, we target p^2/q
        def ais_target_log_prob(x):
            return 2*self.model.target_distribution.log_prob(x) - self.model.flow.log_prob(x)

        self.ais_target_log_prob = ais_target_log_prob  # save
        self.model.annealed_importance_sampler.target_log_prob = ais_target_log_prob


    def run(self,
            n_iterations: int,
            batch_size: int,
            eval_batch_size: Optional[int] = None,
            n_eval: Optional[int] = None,
            n_plot: Optional[int] = None,
            n_checkpoints: Optional[int] = None,
            save: bool = True) -> None:


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

        pbar = tqdm(range(n_iterations))
        for i in pbar:
            self.optimizer.zero_grad()
            # collect samples and log weights with AIS and add to the buffer
            x_ais, log_w_ais = self.model.\
                annealed_importance_sampler.sample_and_log_weights(batch_size)
            x_ais = x_ais.detach()
            log_w_ais = log_w_ais.detach()
            log_q_x = self.model.flow.log_prob(x_ais)
            self.buffer.add(x_ais.detach(), log_w_ais.detach(), log_q_x.detach())

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
                w_adjust = torch.exp(log_q_old - log_q_x).detach() # no grad
                w_adjust = torch.clip(w_adjust, max=self.max_adjust_w_clip)
                # manually calculate the new form of the loss
                loss = - torch.mean(w_adjust * log_q_x)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.max_gradient_norm)
                    self.optimizer.step()
                else:
                    print("nan loss in replay step")
            with torch.no_grad():
                for (x, log_w, log_q_old, indices) in mini_dataset:
                    """Adjust importance weights in the buffer for the points in the `mini_dataset` 
                    to account for the updated theta."""
                    log_q_new = self.model.flow.log_prob(x)
                    log_adjust_insert = log_q_old - log_q_new
                    self.buffer.adjust(log_adjust_insert, log_q_new, indices)

            # we log info from the step of the recently generated ais points.
            info = self.model.get_iter_info()
            info.update(loss=loss.cpu().detach().item(),
                        step=i,
                        grad_norm=grad_norm.cpu().detach().item(),
                        sampled_log_w_std=torch.std(log_w).detach().cpu().item(),
                        sampled_log_w_mean=torch.mean(log_w).detach().cpu().item(),
                        w_adjust_mean=torch.mean(w_adjust).detach().cpu().item(),
                        w_adjust_min=torch.min(w_adjust).detach().cpu().item(),
                        w_adjust_max=torch.max(w_adjust).detach().cpu().item(),
                        log_w_adjust_insert_mean=torch.mean(log_adjust_insert).detach().cpu().item(),
                        log_q_mean=torch.mean(log_q_new).detach().cpu().item()
                        )

            self.logger.write(info)
            pbar.set_description(f"loss: {loss.cpu().detach().item()}, ess base: {info['ess_base']},"
                                 f"ess ais: {info['ess_ais']}")

            if n_eval is not None:
                if i in eval_iter:
                    # set ais distribution to target for evaluation of ess
                    self.model.annealed_importance_sampler.transition_operator.set_eval_mode(False)
                    self.model.annealed_importance_sampler.target_log_prob = self.model.target_distribution.log_prob
                    eval_info_true_target = self.model.get_eval_info(outer_batch_size=eval_batch_size,
                                                         inner_batch_size=batch_size)
                    # set ais distribution back to p^2/q.
                    self.model.annealed_importance_sampler.target_log_prob = self.ais_target_log_prob
                    eval_info_practical_target = self.model.get_eval_info(outer_batch_size=eval_batch_size,
                                                inner_batch_size=batch_size)
                    self.model.annealed_importance_sampler.transition_operator.set_eval_mode(True)
                    eval_info = {}
                    eval_info.update({key + "_p_target": val for key, val in eval_info_true_target.items()})
                    eval_info.update({key + "_p2overq_target": val for key, val in eval_info_practical_target.items()})

                    eval_info.update(step=i)
                    self.logger.write(eval_info)

            if n_plot is not None:
                if i in plot_iter:
                    figures = self.plot(self.model)
                    if save:
                        for j, figure in enumerate(figures):
                            figure.savefig(os.path.join(self.plots_dir, f"{j}_iter_{i}.png"))

            if n_checkpoints is not None:
                if i in checkpoint_iter:
                    checkpoint_path = os.path.join(self.checkpoints_dir, f"iter_{i}/")
                    pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
                    self.model.save(os.path.join(checkpoint_path, "model.pt"))
                    torch.save(self.optimizer.state_dict(),
                               os.path.join(checkpoint_path, 'optimizer.pt'))
                    if self.optim_schedular:
                        torch.save(self.optim_schedular.state_dict(),
                                   os.path.join(self.checkpoints_dir, 'scheduler.pt'))

        self.logger.close()
