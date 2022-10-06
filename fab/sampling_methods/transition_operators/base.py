from typing import Mapping, Any, Callable
import torch

from fab.types_ import LogProbFunc
from fab.sampling_methods.base import Point, get_intermediate_log_prob, \
    get_grad_intermediate_log_prob, create_point


TransitionTargetLogProbFn = Callable[[Point], torch.Tensor]


class TransitionOperator(torch.nn.Module):
    def __init__(self,
                 n_ais_intermediate_distributions: int,
                 dim: int,
                 base_log_prob: LogProbFunc,
                 target_log_prob: LogProbFunc,
                 p_sq_over_q_target: bool,
                 ):
        self.dim = dim
        self.target_log_prob = target_log_prob
        self.base_log_prob = base_log_prob
        self.n_ais_intermediate_distributions = n_ais_intermediate_distributions
        self.p_sq_over_q_target = p_sq_over_q_target
        super(TransitionOperator, self).__init__()


    def create_new_point(self, x: torch.Tensor) -> Point:
        """Create a new point."""
        return create_point(x, self.base_log_prob, self.target_log_prob,
                            with_grad=self.uses_grad_info)


    def intermediate_target_log_prob(self, point: Point, beta: float) -> torch.Tensor:
        return get_intermediate_log_prob(point, beta,
                                         p_sq_over_q_target=self.p_sq_over_q_target)


    def grad_intermediate_target_log_prob(self, point: Point, beta: float) -> torch.Tensor:
        return get_grad_intermediate_log_prob(
            point,
            beta,
            p_sq_over_q_target=self.p_sq_over_q_target
        )

    @property
    def uses_grad_info(self) -> bool:
        """Returns True if the transition operator uses gradient info (e.g. HMC) or False if not
        (e.g. Metropolis)."""
        raise NotImplementedError

    def get_logging_info(self) -> Mapping[str, Any]:
        """Returns a dictionary of relevant information for logging."""
        raise NotImplementedError


    def transition(self, x: Point, i: int, beta: float) -> Point:
        """
        Returns x generated from transition with log_q_x, as the invariant
        distribution.

        Args:
            x: Input samples from the base distribution
            i: Intermediate AIS distribution number.
            beta: Beta controlling interpolation between base and target log prob.

        Returns:
            x: Samples from MCMC with g as the target distribution.
        """
        raise NotImplementedError

    def set_eval_mode(self, eval_setting: bool):
        """Turns on/off any tuning"""
        raise NotImplementedError
