from typing import Tuple, Dict, Any, NamedTuple

import torch
import numpy as np
from functools import partial

from fab.types_ import LogProbFunc
from fab.sampling_methods.transition_operators.base import TransitionOperator
from fab.types_ import Distribution
from fab.utils.numerical import effective_sample_size


class LoggingInfo(NamedTuple):
    ess_base: float
    ess_ais: float
    log_Z: float  # normalisation constant



class AnnealedImportanceSampler:
    def __init__(self,
                 base_distribution: Distribution,
                 target_log_prob: LogProbFunc,
                 transition_operator: TransitionOperator,
                 n_intermediate_distributions: int = 1,
                 distribution_spacing_type: str = "linear"
                 ):
        self.base_distribution = base_distribution
        self.target_log_prob = target_log_prob
        self.transition_operator = transition_operator
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
                               ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Initialise AIS with samples from the base distribution.
        x, log_prob_p0 = self.base_distribution.sample_and_log_prob((batch_size,))
        x, log_prob_p0 = self._remove_nan_and_infs(x, log_prob_p0, descriptor="chain init")
        log_w = self.intermediate_unnormalised_log_prob(x, 1) - log_prob_p0

        # Save effective sample size over samples from base distribution if logging.
        if logging:
            with torch.no_grad():
                log_target_p_x_base_samples = self.target_log_prob(x)
                log_w_base = log_target_p_x_base_samples - log_prob_p0
                ess_base = effective_sample_size(log_w_base).detach().cpu().item()

        # Move through sequence of intermediate distributions via MCMC.
        for j in range(1, self.n_intermediate_distributions+1):
            x, log_w = self.perform_transition(x, log_w, j)

        x, log_w = self._remove_nan_and_infs(x, log_w, descriptor="chain end")

        # Save effective sample size if logging.
        if logging:
            with torch.no_grad():
                ess_ais = effective_sample_size(log_w).cpu().item()
                log_Z_N = torch.logsumexp(log_w, dim=0)
                log_Z = log_Z_N - torch.log(torch.ones_like(log_Z_N) * batch_size)
                self._logging_info = LoggingInfo(ess_base=ess_base, ess_ais=ess_ais,
                                                 log_Z=log_Z.cpu().item())
        return x, log_w


    def perform_transition(self, x_new: torch.Tensor, log_w: torch.Tensor, j: int):
        """"Transition via MCMC with the j'th intermediate distribution as the target."""

        target_p_x = partial(self.intermediate_unnormalised_log_prob, j=j)
        x_new = self.transition_operator.transition(x_new, target_p_x, j-1)
        log_w = log_w + self.intermediate_unnormalised_log_prob(x_new, j + 1) - \
                self.intermediate_unnormalised_log_prob(x_new, j)
        return x_new, log_w


    def intermediate_unnormalised_log_prob(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """Calculate the intermediate log probability density function, by interpolating between
        the base and target distributions log probability density functions."""
        # j is the step of the algorithm, and corresponds which
        # intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta) * self.base_distribution.log_prob(x) + beta * self.target_log_prob(x)


    def setup_distribution_spacing(self, distribution_spacing_type: str,
                                   n_intermediate_distributions: int) -> torch.Tensor:
        """Setup the spacing of the distributions, either with linear or geometric spacing."""
        assert n_intermediate_distributions > 0
        if n_intermediate_distributions < 3:
            print(f"using linear spacing as there are only {n_intermediate_distributions} "
                  f"intermediate distribution")
            distribution_spacing_type = "linear"
        if distribution_spacing_type == "geometric":
            # rough heuristic, copying ratio used in example in AIS paper
            n_linspace_points = max(int(n_intermediate_distributions / 4), 2)
            n_geomspace_points = n_intermediate_distributions - n_linspace_points
            B_space = np.concatenate([np.linspace(0, 0.01, n_linspace_points + 3)[:-1],
                                   np.geomspace(0.01, 1, n_geomspace_points)])
            B_space = np.flip(1 - B_space).copy()
        elif distribution_spacing_type == "linear":
            B_space = np.linspace(0.0, 1.0, n_intermediate_distributions+2)
        else:
            raise Exception(f"distribution spacing incorrectly specified:"
                            f" '{distribution_spacing_type}',"
                            f"options are 'geometric' or 'linear'")
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
            x, log_prob_p0 = self._remove_nan_and_infs(x, log_prob_p0, descriptor="chain init")
            base_log_w = self.target_log_prob(x) - log_prob_p0
            # append base samples and log probs
            base_samples.append(x.detach().cpu())
            base_log_w_s.append(base_log_w.detach().cpu())

            log_w = self.intermediate_unnormalised_log_prob(x, 1) - log_prob_p0
            # Move through sequence of intermediate distributions via MCMC.
            for j in range(1, self.n_intermediate_distributions+1):
                x, log_w = self.perform_transition(x, log_w, j)

            x, log_w = self._remove_nan_and_infs(x, log_w, descriptor="chain end")
            # append ais samples and log probs
            ais_samples.append(x.detach().cpu())
            ais_log_w.append(log_w.detach().cpu())


        base_samples = torch.cat(base_samples, dim=0)
        base_log_w_s = torch.cat(base_log_w_s, dim=0)
        ais_samples = torch.cat(ais_samples, dim=0)
        ais_log_w = torch.cat(ais_log_w, dim=0)

        return base_samples, base_log_w_s, ais_samples, ais_log_w

    def _remove_nan_and_infs(self, x: torch.Tensor, log_p_or_log_w: torch.Tensor,
                             descriptor: str = "chain init") -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Remove any NaN points or log probs / log weights. During the chain initialisation the
        flow can generate Nan/Infs making this function necessary in the first step of AIS. Sometimes
        extreme points may be generated which have NaN probability under the target, which makes
        this function necessary in the final step of AIS."""
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(log_p_or_log_w) & ~torch.isnan(log_p_or_log_w)
        if torch.sum(valid_indices) == 0:  # no valid indices
            raise Exception(f"No valid points generated in sampling the {descriptor}")
        if valid_indices.all():
            pass
        else:
            print(f"{torch.sum(~valid_indices)} nan/inf samples/log-probs/log-weights encountered "
                  f"at {descriptor}.")
            log_p_or_log_w = log_p_or_log_w[valid_indices]
            x = x[valid_indices, :]
        return x, log_p_or_log_w
