from typing import Optional, Dict, Any, Tuple

from fab.types_ import Model
from fab.target_distributions.base import TargetDistribution
from fab.sampling_methods import AnnealedImportanceSampler, HamiltoneanMonteCarlo, \
    TransitionOperator
from fab.trainable_distributions import TrainableDistribution
import torch


class FABModel(Model):
    """Definition of the Flow Annealed Importance Sampling Bootstrap (FAB) model. """
    def __init__(self,
                 flow: TrainableDistribution,
                 target_distribution: TargetDistribution,
                 n_intermediate_distributions: int,
                 transition_operator: Optional[TransitionOperator],
                 ais_distribution_spacing: "str" = "linear",
                 ):
        self.flow = flow
        self.target_distribution = target_distribution
        self.n_intermediate_distributions = n_intermediate_distributions
        assert len(flow.event_shape) == 1, "Currently only 1D distributions are supported"
        if transition_operator is None:
            transition_operator = HamiltoneanMonteCarlo(n_intermediate_distributions,
                                                        flow.event_shape[0])
        self.annealed_importance_sampler = AnnealedImportanceSampler(
            base_distribution=flow,
            target_log_prob=target_distribution.log_prob,
            transition_operator=transition_operator,
            n_intermediate_distributions=n_intermediate_distributions,
            distribution_spacing_type=ais_distribution_spacing)

    def loss(self, batch_size: int) -> torch.Tensor:
        # return self.fab_forward_kl(batch_size)
        return self.fab_alpha_div_loss(batch_size)


    def fab_alpha_div_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        x_ais, log_w_ais = self._remove_nan_and_infs(x_ais, log_w_ais)
        # Estimate log_Z_N where N is the number of samples and Z is the target's normalisation
        # constant to adjust target log probability with. This is to keeping the learning stable
        # (so that we don't have an implicit Z or N constant in the loss that is dependant on the
        # specific target or batch size).
        log_Z_N = torch.logsumexp(log_w_ais, dim=0)
        log_w_AIS_normalised = log_w_ais - log_Z_N
        log_q_x = self.flow.log_prob(x_ais)
        log_p_x = self.target_distribution.log_prob(x_ais)
        # TODO: we may want to investigate using a running average for log_Z to normalise log_p_x.
        # N = log_w_ais.shape[0]
        # log_p_x = self.target_distribution.log_prob(x_ais) - log_Z
        # log_Z = log_Z_N - torch.log(torch.ones_like(log_Z_N) * N)
        log_w = log_p_x - log_q_x
        return torch.logsumexp(log_w_AIS_normalised + log_w, dim=0)


    def fab_forward_kl(self, batch_size: int) -> torch.Tensor:
        """Compute FAB estimate of forward kl-divergence."""
        x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        x_ais, log_w_ais = self._remove_nan_and_infs(x_ais, log_w_ais)
        log_Z_N = torch.logsumexp(log_w_ais, dim=0)
        log_w_AIS_normalised = log_w_ais - log_Z_N
        log_q_x = self.flow.log_prob(x_ais)
        return - torch.sum(torch.exp(log_w_AIS_normalised) * log_q_x)


    def _remove_nan_and_infs(self, x_ais: torch.Tensor, log_w_ais: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # first remove samples that have inf/nan log w
        valid_indices = ~torch.isinf(log_w_ais) & ~torch.isnan(log_w_ais)
        if torch.sum(valid_indices) == 0:  # no valid indices
            raise Exception("No valid importance weights")
        if valid_indices.all():
            pass
        else:
            print(f"{torch.sum(~valid_indices)} nan/inf weights")
            log_w_ais = log_w_ais[valid_indices]
            x_ais = x_ais[valid_indices, :]
        return x_ais, log_w_ais

    def get_iter_info(self) -> Dict[str, Any]:
        return self.annealed_importance_sampler.get_logging_info()

    def get_eval_info(self) -> Dict[str, Any]:
        # TODO: big batch effective sample size, metrics from target.
        raise NotImplementedError



