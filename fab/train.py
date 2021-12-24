from typing import Callable, Any, Optional

import torch.optim.optimizer
from tqdm import tqdm
import numpy as np

from fab.utils.logging import Logger, ListLogger


Model = Any  # TODO needs to be nn.Module with a .loss function
lr_scheduler = Any
Plotter = Callable[[Model], None]

class Trainer:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 optim_schedular: Optional[lr_scheduler],
                 logger: Logger = ListLogger(),
                 plot: Optional[Plotter] = None):
        self.model = model
        self.optimizer = optimizer
        self.optim_schedular = optim_schedular
        self.logger = logger
        self.plot = plot


    def run(self,
            n_iterations: int,
            batch_size: int,
            n_eval: Optional[int] = None,
            n_plot: Optional[int] = None) -> None:
        if n_eval is not None:
            eval_iter = list(np.linspace(0, n_iterations - 1, n_eval, dtype="int"))
        if n_plot is not None:
            plot_iter = list(np.linspace(0, n_iterations - 1, n_plot, dtype="int"))

        pbar = tqdm(range(n_iterations))
        for i in pbar:
            self.optimizer.zero_grad()
            loss = self.model.loss(batch_size)
            info = self.model.get_iter_info()
            info.update(loss=loss.cpu().detach().item())
            self.logger.write(info)
            loss.backward()
            self.optimizer.step()
            if self.optim_schedular:
                self.optim_schedular.step()
            pbar.set_description(f"loss: {loss.cpu().detach().item()}")

            if n_eval is not None:
                if i in eval_iter:
                    eval_info = self.model.eval()
                    self.logger.write(eval_info)

            if n_plot is not None:
                if i in plot_iter:
                    self.plot(self.model)
