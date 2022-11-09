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
                 p_target: bool = True,
                 alpha: float = None,
                 ):
        self.dim = dim
        self.target_log_prob = target_log_prob
        self.base_log_prob = base_log_prob
        self.alpha = alpha
        self.n_ais_intermediate_distributions = n_ais_intermediate_distributions
        self.p_target = p_target
        super(TransitionOperator, self).__init__()


    def create_new_point(self, x: torch.Tensor) -> Point:
        """Create a new instance of a `Point` given an x (sample). See the `Point` definition
        for further details. """
        return create_point(x, self.base_log_prob, self.target_log_prob,
                            with_grad=self.uses_grad_info)


    def intermediate_target_log_prob(self, point: Point, beta: float) -> torch.Tensor:
        with torch.no_grad():
            # We do not backprop through MCMC/AIS. So do not include these gradients.
            return get_intermediate_log_prob(point, beta,
                                             alpha=self.alpha,
                                             p_target=self.p_target)

    def grad_intermediate_target_log_prob(self, point: Point, beta: float) -> torch.Tensor:
        with torch.no_grad():
            # We do not backprop through MCMC/AIS. So do not include second order grads through the
            # intermediate log prob grad.
            return get_grad_intermediate_log_prob(
                point,
                beta,
                alpha=self.alpha,
                p_target=self.p_target
            )

    @property
    def uses_grad_info(self) -> bool:
        """Returns True if the transition operator uses gradient info (e.g. HMC) or False if not
        (e.g. Metropolis)."""
        raise NotImplementedError

    def get_logging_info(self) -> Mapping[str, Any]:
        """Returns a dictionary of relevant information for logging."""
        raise NotImplementedError


    def transition(self, point: Point, i: int, beta: float) -> Point:
        """
        Returns points generated from transition with g, as the invariant
        distribution. g = p^2/q if self.p_sq_over_q_target else g=p, where p is the target
        distribution and q is the trained distribution.

        Args:
            point: Input samples from previous AIS step. Also contains info on the log prob
                of p & q, and if required their gradients.
            i: Intermediate AIS distribution number.
            beta: Beta controlling interpolation between base and target log prob.

        Returns:
            point: Points from MCMC with g as the target distribution.
        """
        raise NotImplementedError

    def set_eval_mode(self, eval_setting: bool):
        """Turns on/off any tuning"""
        raise NotImplementedError
