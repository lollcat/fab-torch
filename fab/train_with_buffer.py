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
    def __init__(self,
                 model: FABModel,
                 optimizer: torch.optim.Optimizer,
                 buffer: ReplayBuffer,
                 n_batches_buffer_sampling: int = 2,
                 optim_schedular: Optional[lr_scheduler] = None,
                 logger: Logger = ListLogger(),
                 plot: Optional[Plotter] = None,
                 gradient_clipping: bool = True,
                 max_gradient_norm: bool = 5.0,
                 save_path: str = ""):
        self.model = model
        self.optimizer = optimizer
        self.optim_schedular = optim_schedular
        self.logger = logger
        self.plot = plot
        self.gradient_clipping = gradient_clipping
        self.max_gradient_norm = max_gradient_norm
        self.save_dir = save_path
        self.plots_dir = os.path.join(self.save_dir, f"plots")
        self.checkpoints_dir = os.path.join(self.save_dir, f"model_checkpoints")
        self.buffer = buffer
        self.n_batches_buffer_sampling = n_batches_buffer_sampling


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
            x_ais, log_w_ais = self.model.\
                annealed_importance_sampler.sample_and_log_weights(batch_size)
            x_ais = x_ais.detach()
            log_w_ais = log_w_ais.detach()
            loss = self.model.fab_alpha_div_loss_inner(x_ais, log_w_ais)
            loss.backward()
            if self.gradient_clipping:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           self.max_gradient_norm)
            self.optimizer.step()
            if self.optim_schedular:
                self.optim_schedular.step()

            # We now take an additinal self.n_batches_buffer_sampling gradient steps using
            # data from the replay buffer.
            for (x, log_w) in self.buffer.sample_n_batches(batch_size,
                                                           self.n_batches_buffer_sampling):
                x, log_w = x.to(self.model.flow.device), log_w.to(self.model.flow.device)
                loss = self.model.fab_alpha_div_loss_inner(x, log_w)
                loss.backward()
                if self.gradient_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.max_gradient_norm)
                self.optimizer.step()
                if self.optim_schedular:
                    self.optim_schedular.step()

            # add data to buffer
            self.buffer.add(x_ais, log_w_ais)


            info = self.model.get_iter_info()
            info.update(loss=loss.cpu().detach().item(),
                        step=i)
            if self.gradient_clipping:
                info.update(grad_norm=grad_norm.cpu().detach().item())
            self.logger.write(info)
            pbar.set_description(f"loss: {loss.cpu().detach().item()}, ess base: {info['ess_base']},"
                                 f"ess ais: {info['ess_ais']}")

            if n_eval is not None:
                if i in eval_iter:
                    eval_info = self.model.get_eval_info(outer_batch_size=eval_batch_size,
                                                inner_batch_size=batch_size)
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
