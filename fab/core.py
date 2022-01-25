from typing import Optional, Dict, Any, Tuple
import torch

from fab.types_ import Model
from fab.target_distributions.base import TargetDistribution
from fab.sampling_methods import AnnealedImportanceSampler, HamiltoneanMonteCarlo, \
    TransitionOperator
from fab.trainable_distributions import TrainableDistribution
from fab.utils.numerical import effective_sample_size



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

    def parameters(self):
        return self.flow.parameters()

    def loss(self, batch_size: int) -> torch.Tensor:
        # return self.fab_forward_kl(batch_size)
        return self.fab_alpha_div_loss(batch_size)


    def fab_alpha_div_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        # Estimate log_Z_N where N is the number of samples and Z is the target's normalisation
        # constant to adjust target log probability with. This is to keeping the learning stable
        # (so that we don't have an implicit Z or N constant in the loss that is dependant on the
        # specific target or batch size).
        log_Z_N = torch.logsumexp(log_w_ais, dim=0)
        log_w_AIS_normalised = log_w_ais - log_Z_N
        log_q_x = self.flow.log_prob(x_ais)
        log_p_x = self.target_distribution.log_prob(x_ais)
        log_w_normalised = log_p_x - log_q_x - log_Z_N
        return torch.logsumexp(log_w_AIS_normalised + log_w_normalised, dim=0)


    def fab_forward_kl(self, batch_size: int) -> torch.Tensor:
        """Compute FAB estimate of forward kl-divergence."""
        x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        log_Z_N = torch.logsumexp(log_w_ais, dim=0)
        log_w_AIS_normalised = log_w_ais - log_Z_N
        log_q_x = self.flow.log_prob(x_ais)
        return - torch.mean(torch.exp(log_w_AIS_normalised) * log_q_x)

    def get_iter_info(self) -> Dict[str, Any]:
        return self.annealed_importance_sampler.get_logging_info()

    def get_eval_info(self,
                      outer_batch_size: int,
                      inner_batch_size: int,
                      ) -> Dict[str, Any]:
        base_samples, base_log_w, ais_samples, ais_log_w = \
            self.annealed_importance_sampler.generate_eval_data(outer_batch_size, inner_batch_size)
        info = {"eval_ess_flow": effective_sample_size(log_w=base_log_w, normalised=False),
                "eval_ess_ais": effective_sample_size(log_w=ais_log_w, normalised=False)}
        flow_info = self.target_distribution.performance_metrics(base_samples, base_log_w,
                                                                 self.flow.log_prob)
        ais_info = self.target_distribution.performance_metrics(ais_samples, ais_log_w)
        info.update(flow_info)
        info.update(ais_info)
        return info
