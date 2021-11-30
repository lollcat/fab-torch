from fab.target_distributions.base import TargetDistribution
from fab.sampling_methods import AnnealedImportanceSampler, HamiltoneanMonteCarlo, \
    TransitionOperator
from fab.learnt_distributions import LearntDistribubtion
import torch

class FABModel:
    """Definition of the Flow Annealed Importance Sampling Bootstrap (FAB) model. """
    def __init__(self,
                 flow: LearntDistribubtion,
                 target_distribution: TargetDistribution,
                 n_intermediate_distributions: int,
                 transition_operator: TransitionOperator = HamiltoneanMonteCarlo,
                 ais_distribution_spacing: "str" = "linear",
                 ):
        self.flow = flow
        self.target_distribution = target_distribution
        self.n_intermediate_distributions = n_intermediate_distributions
        self.annealed_importance_sampler = AnnealedImportanceSampler(
            base_distribution=flow,
            target_log_prob=target_distribution.log_prob,
            transition_operator=transition_operator,
            n_intermediate_distributions=n_intermediate_distributions,
            distribution_spacing_type=ais_distribution_spacing)


    def fab_loss(self, batch_size: int) -> torch.Tensor:
        x_AIS, log_w_AIS = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_AIS = x_AIS.detach()
        log_w_AIS = log_w_AIS.detach()
        log_w_AIS_normalised = log_w_AIS - torch.logsumexp(log_w_AIS, dim=0)
        log_q_x = self.flow.log_prob(x_AIS)
        log_p_x = self.target_distribution.log_prob(x_AIS)
        log_w = log_p_x - log_q_x
        return torch.logsumexp(log_w_AIS_normalised + log_w, dim=0)
