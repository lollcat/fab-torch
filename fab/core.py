from typing import Optional, Dict, Any
import torch
import warnings

from fab.types_ import Model
from fab.target_distributions.base import TargetDistribution
from fab.sampling_methods import AnnealedImportanceSampler, HamiltonianMonteCarlo, \
    TransitionOperator, Point
from fab.trainable_distributions import TrainableDistribution
from fab.utils.numerical import effective_sample_size


P_SQ_OVER_Q_TARGET_LOSSES = ["p2_over_q_alpha_2_div"]
LOSSES_USING_AIS = ["p2_over_q_alpha_2_div", "lb_alpha_2_div", "sample_log_prob"]

class FABModel(Model):
    """Definition of the Flow Annealed Importance Sampling Bootstrap (FAB) model. """
    def __init__(self,
                 flow: TrainableDistribution,
                 target_distribution: TargetDistribution,
                 n_intermediate_distributions: int,
                 transition_operator: Optional[TransitionOperator] = None,
                 ais_distribution_spacing: "str" = "linear",
                 loss_type: "str" = "lb_alpha_2_div",
                 ):
        assert loss_type in ["lb_alpha_2_div", "forward_kl", "sample_log_prob",
                             "flow_forward_kl", "flow_alpha_2_div",
                             "flow_reverse_kl", "p2_over_q_alpha_2_div",
                             "flow_alpha_2_div_unbiased", "flow_alpha_2_div_nis",
                             "target_forward_kl"]
        self.p_sq_over_q_target = loss_type in P_SQ_OVER_Q_TARGET_LOSSES
        self.loss_type = loss_type
        self.flow = flow
        self.target_distribution = target_distribution
        self.n_intermediate_distributions = n_intermediate_distributions
        self.ais_distribution_spacing = ais_distribution_spacing
        assert len(flow.event_shape) == 1, "Currently only 1D distributions are supported"
        if loss_type in LOSSES_USING_AIS:
            self.transition_operator = transition_operator
            self.annealed_importance_sampler = AnnealedImportanceSampler(
                base_distribution=self.flow,
                target_log_prob=self.target_distribution.log_prob,
                transition_operator=self.transition_operator,
                n_intermediate_distributions=self.n_intermediate_distributions,
                distribution_spacing_type=self.ais_distribution_spacing,
                p_sq_over_q_target=self.p_sq_over_q_target
            )

    def parameters(self):
        return self.flow.parameters()

    def loss(self, args) -> torch.Tensor:
        if self.loss_type == "lb_alpha_2_div":
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
        elif self.loss_type == "flow_alpha_2_div_unbiased":
            return self.flow_alpha_2_div_unbiased(args)
        elif self.loss_type == "p2_over_q_alpha_2_div":
            return self.p2_over_q_alpha_2_div(args)
        elif self.loss_type == "flow_alpha_2_div_nis":
            return self.flow_alpha_2_div_nis(args)
        elif self.loss_type == "target_forward_kl":
            return self.target_forward_kl(args)
        else:
            raise NotImplementedError

    def set_ais_target(self, target: str = "p"):
        if target == "p":
            self.annealed_importance_sampler.p_sq_over_q_target = False
            self.annealed_importance_sampler.transition_operator.p_sq_over_q_target = False
        else:
            self.annealed_importance_sampler.p_sq_over_q_target = True
            self.annealed_importance_sampler.transition_operator.p_sq_over_q_target = True

    def flow_reverse_kl(self, batch_size: int) -> torch.Tensor:
        x, log_q = self.flow.sample_and_log_prob((batch_size,))
        log_p = self.target_distribution.log_prob(x)
        return torch.mean(log_q) - torch.mean(log_p)

    def flow_alpha_2_div(self, batch_size: int) -> torch.Tensor:
        x, log_q = self.flow.sample_and_log_prob((batch_size,))
        log_p = self.target_distribution.log_prob(x)
        return torch.logsumexp(2 * (log_p - log_q), 0)

    def flow_alpha_2_div_unbiased(self, batch_size: int) -> torch.Tensor:
        """Compute an unbiased estimate of alpha-2-divergence with samples from the flow."""
        x, log_q_x = self.flow.sample_and_log_prob((batch_size,))
        log_p_x = self.target_distribution.log_prob(x)
        loss = torch.mean(torch.exp(2*(log_p_x - log_q_x)) * log_q_x)
        return loss

    def flow_alpha_2_div_nis(self, batch_size: int) -> torch.Tensor:
        """From Neural Importance sampling paper https://arxiv.org/pdf/1808.03856.pdf."""
        x, log_q_x = self.flow.sample_and_log_prob((batch_size,))
        log_p_x = self.target_distribution.log_prob(x)
        loss = - torch.mean(torch.exp(2*(log_p_x - log_q_x)).detach() * log_q_x)
        return loss

    def p2_over_q_alpha_2_div(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss with p^2/q as the AIS target."""
        # set ais target distribution to p^2/q
        self.set_ais_target("p^2/q")
        if isinstance(self.annealed_importance_sampler.transition_operator, HamiltonianMonteCarlo):
            point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        else:
            with torch.no_grad():
                point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(
                    batch_size)
        log_w_ais = log_w_ais.detach()
        log_q_x = point_ais.log_q
        loss = - torch.mean(torch.softmax(log_w_ais, dim=-1) * log_q_x)
        # reset ais target distribution back to p
        self.set_ais_target("p")
        return loss

    def inner_loss(self, point: Point, log_w_ais) -> torch.Tensor:
        """Loss as a function of ais samples and weights, we use this when training with a replay buffer."""
        if self.loss_type == "lb_alpha_2_div":
            return self.fab_lb_alpha_div_loss_inner(point.log_q, point.log_q, log_w_ais)
        elif self.loss_type == "forward_kl":
            return self.fab_forward_kl_inner(point.log_q, log_w_ais)
        else:
            raise NotImplementedError

    def fab_lb_alpha_div_loss_inner(self, log_q, log_p, log_w_ais) -> torch.Tensor:
        """Compute the FAB loss based on upper-bound of alpha-divergence with alpha=2 from
        https://arxiv.org/abs/2111.11510."""
        log_w = log_p - log_q
        return torch.logsumexp(log_w_ais + log_w, dim=0)

    def fab_alpha_div_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        loss = self.fab_lb_alpha_div_loss_inner(point_ais.log_q, point_ais.log_p, log_w_ais)
        return loss

    def target_forward_kl(self, batch_size: int) -> torch.Tensor:
        """Assumes we can sample from the target distribution"""
        x = self.target_distribution.sample((batch_size,))
        return self.flow_forward_kl(x)

    def flow_forward_kl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward KL-divergence of flow"""
        return -torch.mean(self.flow.log_prob(x))

    def fab_forward_kl_inner(self, log_q_x, log_w_ais) -> torch.Tensor:
        w_ais = torch.softmax(log_w_ais, dim=0)
        return - torch.mean(w_ais * log_q_x)

    def fab_forward_kl(self, batch_size: int) -> torch.Tensor:
        """Compute FAB estimate of forward kl-divergence."""
        point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        log_w_ais = log_w_ais.detach()
        return self.fab_forward_kl_inner(point_ais.log_q, log_w_ais)

    def fab_sample_log_prob(self, batch_size: int, sample_frac: float = 1.0) -> torch.Tensor:
        """Compute FAB loss by maximising the log prob of ais samples under the flow."""
        point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        return - torch.mean(point_ais.log_q)

    def get_iter_info(self) -> Dict[str, Any]:
        if hasattr(self, "annealed_importance_sampler"):
            return self.annealed_importance_sampler.get_logging_info()
        else:
            return {}

    def get_eval_info(self,
                      outer_batch_size: int,
                      inner_batch_size: int,
                      ) -> Dict[str, Any]:
        if hasattr(self, "annealed_importance_sampler"):
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
        else:
            raise NotImplementedError
            # TODO
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
            # If flow is incorretly loaded then this will mess up evaluation, so raise Error.
            raise RuntimeError('Flow could not be loaded. '
                  'Perhaps there is a mismatch in the architectures.')
        try:
            self.transition_operator.load_state_dict(checkpoint['trans_op'])
        except RuntimeError:
            # Sometimes we only evaluate the flow, in which case having a transition operator
            # mismatch is okay, so we raise a warning.
            warnings.warn('Transition operator could not be loaded. '
                  'Perhaps there is a mismatch in the architectures.')
        self.annealed_importance_sampler = AnnealedImportanceSampler(
            base_distribution=self.flow,
            target_log_prob=self.target_distribution.log_prob,
            transition_operator=self.transition_operator,
            p_sq_over_q_target=self.p_sq_over_q_target,
            n_intermediate_distributions=self.n_intermediate_distributions,
            distribution_spacing_type=self.ais_distribution_spacing)
