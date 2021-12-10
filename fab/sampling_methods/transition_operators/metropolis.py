from typing import Dict

import torch
import torch.nn as nn

from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.types import LogProbFunc

class Metropolis(nn.Module, TransitionOperator):
    def __init__(self, n_transitions, n_updates, max_step_size=1.0, min_step_size=0.1,
                 adjust_step_size=True, target_p_accept=0.1):
        """
        Designed for use in annealed importance sampler
        :param n_updates: number of metropolis updates
        :param noise_scalings: can be sequence e.g. e.g. tensor([2.0, 1.0, 0.1])
        """
        super(Metropolis, self).__init__()
        self.n_distributions = n_transitions
        self.n_updates = n_updates
        self.adjust_step_size = adjust_step_size
        self.register_buffer("noise_scaling_ratios", torch.linspace(max_step_size, min_step_size,
                                                                    n_updates).repeat(
            (n_transitions, 1)))
        self.target_prob_accept = target_p_accept

    def get_logging_info(self) -> Dict:
        """Return the first and last noise scaling size for logging."""
        interesting_dict = {}
        interesting_dict[f"noise_scaling_0_0"] = self.noise_scaling_ratios[0, 0].cpu().item()
        interesting_dict[f"noise_scaling_0_-1"] = self.noise_scaling_ratios[0, -1].cpu().item()
        return interesting_dict


    def transition(self, x: torch.Tensor, log_p_x: LogProbFunc, i: int) -> torch.Tensor:
        """Returns x generated from transition with log_p_x using the Metropolis algorithm."""
        for n in range(self.n_updates):
            x_proposed = x + torch.randn(x.shape).to(x.device) * self.noise_scalings[i, n]
            x_proposed_log_prob = log_p_x(x_proposed)
            x_prev_log_prob = log_p_x(x)
            acceptance_probability = torch.exp(x_proposed_log_prob - x_prev_log_prob)
            # not that sometimes this will be greater than one, corresonding to 100% probability of
            # acceptance
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape
                                                          ).to(x.device)).int()
            accept = accept[:, None].repeat(1, x.shape[-1])
            x = accept * x_proposed + (1 - accept) * x
            if self.auto_adjust:
                p_accept = torch.mean(torch.clamp_max(acceptance_probability, 1))
                if p_accept > self.target_prob_accept:  # too much accept
                    self.noise_scaling_ratios[i, n] = self.noise_scaling_ratios[i, n] * 1.1
                else:
                    self.noise_scaling_ratios[i, n] = self.noise_scaling_ratios[i, n] * 0.9
        return x