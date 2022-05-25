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
                 loss_type: "str" = "alpha_2_div",
                 ):
        assert loss_type in ["alpha_2_div", "forward_kl", "sample_log_prob",
                             "flow_forward_kl", "flow_alpha_2_div",
                             "flow_reverse_kl"]
        self.loss_type = loss_type
        self.flow = flow
        self.target_distribution = target_distribution
        self.n_intermediate_distributions = n_intermediate_distributions
        self.ais_distribution_spacing = ais_distribution_spacing
        assert len(flow.event_shape) == 1, "Currently only 1D distributions are supported"
        if transition_operator is None:
            self.transition_operator = HamiltoneanMonteCarlo(self.n_intermediate_distributions,
                                                             self.flow.event_shape[0])
        else:
            self.transition_operator = transition_operator
        self.annealed_importance_sampler = AnnealedImportanceSampler(
            base_distribution=self.flow,
            target_log_prob=self.target_distribution.log_prob,
            transition_operator=self.transition_operator,
            n_intermediate_distributions=self.n_intermediate_distributions,
            distribution_spacing_type=self.ais_distribution_spacing)

    def parameters(self):
        return self.flow.parameters()

    def loss(self, args) -> torch.Tensor:
        if self.loss_type == "alpha_2_div":
            return self.fab_alpha_div_loss(args)
        elif self.loss_type == "forward_kl":
            return self.fab_forward_kl(args)
        elif self.loss_type == "sample_log_prob":
            return self.fab_sample_log_prob(args)
        elif self.loss_type == "flow_forward_kl":
            return self.flow_forward_kl(args)
        elif self.loss_type == "flow_reverse_kl":
            return self.flow_reverse_kl(args)
        elif self.loss_type == "flow_alpha_2_div":
            return self.flow_alpha_2_div(args)
        else:
            raise NotImplementedError

    def flow_reverse_kl(self, batch_size: int) -> torch.Tensor:
        x, log_q = self.flow.sample_and_log_prob((batch_size,))
        log_p = self.target_distribution.log_prob(x)
        return torch.mean(log_q) - torch.mean(log_p)

    def flow_alpha_2_div(self, batch_size: int) -> torch.Tensor:
        x, log_q = self.flow.sample_and_log_prob((batch_size,))
        log_p = self.target_distribution.log_prob(x)
        return -torch.logsumexp(2 * (log_p - log_q), 0)

    def fab_alpha_div_loss_inner(self, x_ais, log_w_ais) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        log_q_x = self.flow.log_prob(x_ais)
        log_p_x = self.target_distribution.log_prob(x_ais)
        log_w = log_p_x - log_q_x
        return torch.logsumexp(log_w_ais + log_w, dim=0)

    def fab_alpha_div_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        if isinstance(self.annealed_importance_sampler.transition_operator, HamiltoneanMonteCarlo):
            x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        else:
            with torch.no_grad():
                x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        loss = self.fab_alpha_div_loss_inner(x_ais, log_w_ais)
        return loss

    def flow_forward_kl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward KL-divergence of flow"""
        return -torch.mean(self.flow.log_prob(x))

    def fab_forward_kl(self, batch_size: int) -> torch.Tensor:
        """Compute FAB estimate of forward kl-divergence."""
        if isinstance(self.annealed_importance_sampler.transition_operator, HamiltoneanMonteCarlo):
            x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        else:
            with torch.no_grad():
                x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        w_ais = torch.softmax(log_w_ais, dim=0)
        log_q_x = self.flow.log_prob(x_ais)
        return - torch.mean(w_ais * log_q_x)

    def fab_sample_log_prob(self, batch_size: int, sample_frac: float = 1.0) -> torch.Tensor:
        """Compute FAB loss by maximising the log prob of ais samples under the flow."""
        if isinstance(self.annealed_importance_sampler.transition_operator, HamiltoneanMonteCarlo):
            x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        else:
            with torch.no_grad():
                x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        log_q_x = self.flow.log_prob(x_ais.detach())
        return - torch.mean(log_q_x)

    def get_iter_info(self) -> Dict[str, Any]:
        return self.annealed_importance_sampler.get_logging_info()

    def get_eval_info(self,
                      outer_batch_size: int,
                      inner_batch_size: int,
                      ) -> Dict[str, Any]:
        base_samples, base_log_w, ais_samples, ais_log_w = \
            self.annealed_importance_sampler.generate_eval_data(outer_batch_size, inner_batch_size)
        info = {"eval_ess_flow": effective_sample_size(log_w=base_log_w, normalised=False).item(),
                "eval_ess_ais": effective_sample_size(log_w=ais_log_w, normalised=False).item()}
        flow_info = self.target_distribution.performance_metrics(base_samples, base_log_w,
                                                                 self.flow.log_prob,
                                                                 batch_size=inner_batch_size)
        ais_info = self.target_distribution.performance_metrics(ais_samples, ais_log_w)
        info.update(flow_info)
        info.update(ais_info)
        return info

    def save(self,
             path: "str"
             ):
        """Save FAB model to file."""
        torch.save({'flow': self.flow._nf_model.state_dict(),
                    'trans_op': self.transition_operator.state_dict()},
                   path)

    def load(self,
             path: "str",
             map_location: Optional[str] = None,
             ):
        """Load FAB model from file."""
        checkpoint = torch.load(path, map_location=map_location)
        try:
            self.flow._nf_model.load_state_dict(checkpoint['flow'])
        except RuntimeError:
            print('Flow could not be loaded. '
                  'Perhaps there is a mismatch in the architectures.')
        try:
            self.transition_operator.load_state_dict(checkpoint['trans_op'])
        except RuntimeError:
            print('Transition operator could not be loaded. '
                  'Perhaps there is a mismatch in the architectures.')
        self.annealed_importance_sampler = AnnealedImportanceSampler(
            base_distribution=self.flow,
            target_log_prob=self.target_distribution.log_prob,
            transition_operator=self.transition_operator,
            n_intermediate_distributions=self.n_intermediate_distributions,
            distribution_spacing_type=self.ais_distribution_spacing)

