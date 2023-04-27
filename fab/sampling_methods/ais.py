from typing import Tuple, Dict, Any, NamedTuple, Optional

import torch
import numpy as np

from fab.types_ import LogProbFunc
from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.types_ import Distribution
from fab.utils.numerical import effective_sample_size
from fab.sampling_methods.base import get_intermediate_log_prob, create_point, Point


class LoggingInfo(NamedTuple):
    ess_base: float
    ess_ais: float
    log_Z: float  # normalisation constant



class AnnealedImportanceSampler:
    """Runs annealed importance sampling. Designed for use with FAB."""
    def __init__(self,
                 base_distribution: Distribution,
                 target_log_prob: LogProbFunc,
                 transition_operator: TransitionOperator,
                 p_target: bool,
                 alpha: Optional[float] = None,
                 n_intermediate_distributions: int = 1,
                 distribution_spacing_type: str = "linear"
                 ):
        if not p_target:
            assert alpha is not None, "Must specify alpha if AIS target is not p."
        self.base_distribution = base_distribution
        self.target_log_prob = target_log_prob
        self.transition_operator = transition_operator
        self.p_target = p_target
        self.alpha = alpha
        self.n_intermediate_distributions = n_intermediate_distributions
        self.distribution_spacing_type = distribution_spacing_type
        self.B_space = self.setup_distribution_spacing(distribution_spacing_type,
                                                       n_intermediate_distributions)
        self._logging_info: LoggingInfo


    def get_logging_info(self) -> Dict[str, Any]:
        """Return information saved during the last call to sample_and_log_weights (assuming
        logging was set to True)."""
        logging_info = self._logging_info._asdict()
        logging_info.update(self.transition_operator.get_logging_info())
        return logging_info


    def sample_and_log_weights(self, batch_size: int, logging: bool = True,
                               ) -> Tuple[Point, torch.Tensor]:
        # Initialise AIS with samples from the base distribution.
        x, log_prob_p0 = self.base_distribution.sample_and_log_prob((batch_size,))
        point = create_point(x,
                             self.base_distribution.log_prob,
                             self.target_log_prob,
                             with_grad=self.transition_operator.uses_grad_info,
                             log_q_x=log_prob_p0
                             )
        log_w = get_intermediate_log_prob(
            point, self.B_space[1], self.alpha, self.p_target) - log_prob_p0
        point, log_w = self._remove_nan_and_infs(point, log_w, descriptor="chain init")

        # Save effective sample size over samples from base distribution if logging.
        if logging:
            with torch.no_grad():
                log_w_base = point.log_p - point.log_q
                ess_base = effective_sample_size(log_w_base).detach().cpu().item()

        # Move through sequence of intermediate distributions via MCMC.
        for j in range(1, self.n_intermediate_distributions+1):
            point, log_w = self.perform_transition(point, log_w, j)

        point, log_w = self._remove_nan_and_infs(point, log_w, descriptor="chain end")

        # Save effective sample size if logging.
        if logging:
            with torch.no_grad():
                ess_ais = effective_sample_size(log_w).cpu().item()
                log_Z_N = torch.logsumexp(log_w, dim=0)
                log_Z = log_Z_N - torch.log(torch.ones_like(log_Z_N) * batch_size)
                self._logging_info = LoggingInfo(ess_base=ess_base, ess_ais=ess_ais,
                                                 log_Z=log_Z.cpu().item())
        return point, log_w.detach()


    def perform_transition(self, x_new: Point, log_w: torch.Tensor, j: int):
        """" Transition via MCMC with the j'th intermediate distribution as the target."""
        x_new = self.transition_operator.transition(x_new, j, self.B_space[j])
        if self.B_space[j + 1] != self.B_space[j]:
            log_numerator = get_intermediate_log_prob(x_new,
                                                self.B_space[j + 1], self.alpha,
                                                self.p_target)
            log_denominator = get_intermediate_log_prob(x_new, self.B_space[j], self.alpha,
                                                self.p_target)
            log_w_increment = log_numerator - log_denominator
            log_w = log_w + log_w_increment
        else:
            # Commonly we may have a few transitions with beta=1 at the end of AIS, which does not
            # change the AIS weights.
            pass
        return x_new, log_w


    def setup_distribution_spacing(self, distribution_spacing_type: str,
                                   n_intermediate_distributions: int) -> torch.Tensor:
        """Setup the spacing of the distributions, either with linear or geometric spacing."""
        assert n_intermediate_distributions > 0
        if distribution_spacing_type == "geometric":
            # rough heuristic, copying ratio used in example in AIS paper.
            # One quarter of Beta linearly spaced between 0 and 0.01
            n_intermediate_linspace_points = int(n_intermediate_distributions / 4)
            # The rest geometrically spaced between 0.01 and 1.0
            n_intermediate_geomspace_points = n_intermediate_distributions - \
                                              n_intermediate_linspace_points - 1
            B_space = np.concatenate([np.linspace(0, 0.01, n_intermediate_linspace_points + 2)[:-1],
                                   np.geomspace(0.01, 1, n_intermediate_geomspace_points + 2)])
        elif distribution_spacing_type == "linear":
            B_space = np.linspace(0.0, 1.0, n_intermediate_distributions+2)
        else:
            raise Exception(f"distribution spacing incorrectly specified:"
                            f" '{distribution_spacing_type}',"
                            f"options are 'geometric' or 'linear'")

        assert B_space.shape == (self.n_intermediate_distributions + 2,)
        return torch.tensor(B_space)


    def generate_eval_data(self, outer_batch_size: int, inner_batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a big batch of data for evaluate, by running multiple steps forward passes of
        AIS. This prevents the GPU getting overloaded.
        Args:
            outer_batch_size: Total number of evaluation points generated.
            inner_batch_size: Batch size during each forward pass of ais.

        Returns:
            base_samples: Samples from the base (flow) distribution.
            base_log_w: Log importance weights for samples from the base distribution.
            ais_samples: Samples from AIS.
            ais_log_w: Log importance weights from AIS.
        """
        base_samples = []
        base_log_w_s = []
        ais_samples = []
        ais_log_w = []
        assert outer_batch_size % inner_batch_size == 0
        n_batches = outer_batch_size // inner_batch_size
        for i in range(n_batches):
            # Initialise AIS with samples from the base distribution.
            x, log_prob_p0 = self.base_distribution.sample_and_log_prob((inner_batch_size,))
            point = create_point(x,
                                 self.base_distribution.log_prob,
                                 self.target_log_prob,
                                 with_grad=self.transition_operator.uses_grad_info,
                                 log_q_x=log_prob_p0
                                 )
            base_log_w = self.target_log_prob(x) - log_prob_p0
            point, base_log_w = self._remove_nan_and_infs(point, base_log_w,
                                                          descriptor="chain init")

            # append base samples and log probs
            base_samples.append(point.x.detach().cpu())
            base_log_w_s.append(base_log_w.detach().cpu())

            log_w = get_intermediate_log_prob(point, self.B_space[1], self.alpha,
                                              self.p_target) - point.log_q
            # Move through sequence of intermediate distributions via MCMC.
            for j in range(1, self.n_intermediate_distributions+1):
                point, log_w = self.perform_transition(point, log_w, j)

            point, log_w = self._remove_nan_and_infs(point, log_w, descriptor="chain end",
                                                     raise_exception=False)
            # append ais samples and log probs
            ais_samples.append(point.x.detach().cpu())
            ais_log_w.append(log_w.detach().cpu())


        base_samples = torch.cat(base_samples, dim=0)
        base_log_w_s = torch.cat(base_log_w_s, dim=0)
        ais_samples = torch.cat(ais_samples, dim=0)
        ais_log_w = torch.cat(ais_log_w, dim=0)

        return base_samples, base_log_w_s, ais_samples, ais_log_w

    def _remove_nan_and_infs(self,
                             point: Point,
                             log_w: torch.Tensor,
                             descriptor: str = "chain init",
                             raise_exception: bool = True) -> Tuple[Point, torch.Tensor]:
        """Remove any NaN points or log probs / log weights. During the chain initialisation the
        flow can generate Nan/Infs making this function necessary in the first step of AIS. Sometimes
        extreme points may be generated which have NaN probability under the target, which makes
        this function necessary in the final step of AIS."""
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(point.log_p) & ~torch.isnan(point.log_p) & \
                        ~torch.isinf(point.log_q) & ~torch.isnan(point.log_q)
        if torch.sum(valid_indices) == 0:  # no valid indices
            if raise_exception:
                raise Exception(f"No valid points generated in sampling the {descriptor}")
            else:
                print(f"No valid points generated in sampling the {descriptor}")
                return point, log_w
        if valid_indices.all():
            pass
        else:
            print(f"{torch.sum(~valid_indices)} nan/inf samples/log-probs/log-weights encountered "
                  f"at {descriptor}.")
        return point[valid_indices], log_w[valid_indices]
