from typing import Callable, Any, Optional, List

import torch.optim.optimizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from fab.utils.logging import Logger, ListLogger
from fab.core import FABModel
from fab.utils.replay_buffer import ReplayBuffer


lr_scheduler = Any  # a learning rate schedular from torch.optim.lr_scheduler
Plotter = Callable[[FABModel], List[plt.Figure]]

class BufferTrainer:
    """A trainer for the FABModel for use with a uniform replay buffer."""
    def __init__(self,
                 model: FABModel,
                 optimizer: torch.optim.Optimizer,
                 buffer: ReplayBuffer,
                 n_batches_buffer_sampling: int = 2,
                 optim_schedular: Optional[lr_scheduler] = None,
                 logger: Logger = ListLogger(),
                 plot: Optional[Plotter] = None,
                 max_gradient_norm: Optional[float] = 5.0,
                 save_path: str = "",
                 clip_ais_weights_frac: Optional[float] = None):
        raise Exception("This code is experimental and has not been updated in a while")
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
        self.clip_ais_weights_frac = clip_ais_weights_frac


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
            # collect samples and log weights with AIS.
            x_ais, log_w_ais = self.model.\
                annealed_importance_sampler.sample_and_log_weights(batch_size)
            x_ais = x_ais.detach()
            log_w_ais = log_w_ais.detach()
            if self.clip_ais_weights_frac is not None:
                # optional clipping of log weights
                k = max(2, int(self.clip_ais_weights_frac * log_w_ais.shape[0]))
                max_log_w = torch.min(torch.topk(log_w_ais, k, dim=0).values)
                log_w_ais = torch.clamp_max(log_w_ais, max_log_w)

            # perform one update using the recently collected AIS samples and log weights
            loss = self.model.inner_loss(x_ais, log_w_ais)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           self.max_gradient_norm)
                self.optimizer.step()
                if self.optim_schedular:
                    self.optim_schedular.step()
            else:
                print("nan loss in non-replay step")

            # we log info from the step of the recently generated ais points.
            info = self.model.get_iter_info()
            info.update(loss=loss.cpu().detach().item(),
                        step=i,
                        grad_norm=grad_norm.cpu().detach().item())
            self.logger.write(info)

            # We now take an additional self.n_batches_buffer_sampling gradient steps using
            # data from the replay buffer.
            for (x, log_w) in self.buffer.sample_n_batches(
                    batch_size=batch_size, n_batches=self.n_batches_buffer_sampling):

                x, log_w = x.to(self.flow_device), log_w.to(self.flow_device)
                self.optimizer.zero_grad()
                loss = self.model.inner_loss(x, log_w)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.max_gradient_norm)
                    self.optimizer.step()
                else:
                    print("nan loss in replay step")

            # add data to buffer
            self.buffer.add(x_ais, log_w_ais)
            pbar.set_description(f"loss: {loss.cpu().detach().item()}, ess base: {info['ess_base']},"
                                 f"ess ais: {info['ess_ais']}")

            if n_eval is not None:
                if i in eval_iter:
                    self.model.annealed_importance_sampler.transition_operator.set_eval_mode(True)
                    eval_info = self.model.get_eval_info(outer_batch_size=eval_batch_size,
                                                inner_batch_size=batch_size)
                    eval_info.update(step=i)
                    self.logger.write(eval_info)
                    self.model.annealed_importance_sampler.transition_operator.set_eval_mode(False)

            if n_plot is not None:
                if i in plot_iter:
                    figures = self.plot(self.model)
                    for j, figure in enumerate(figures):
                        if save:
                            figure.savefig(os.path.join(self.plots_dir, f"{j}_iter_{i}.png"))
                        plt.close(figure)


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
