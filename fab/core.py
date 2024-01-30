from typing import Optional, Dict, Any, Union
import torch
import numpy as np
import warnings

from fab.types_ import Model
from fab.target_distributions.base import TargetDistribution
from fab.sampling_methods import AnnealedImportanceSampler, TransitionOperator, Point
from fab.trainable_distributions import TrainableDistribution
from fab.utils.numerical import effective_sample_size


ALPHA_DIV_TARGET_LOSSES = ["fab_alpha_div"]
LOSSES_USING_AIS = ["fab_alpha_div", "fab_ub_alpha_2_div", None]
EXPERIMENTAL_LOSSES = ["flow_alpha_2_div_unbiased", "flow_alpha_2_div", "fab_ub_alpha_2_div"]


class FABModel(Model):
    """Definition of various models, including the Flow Annealed Importance Sampling Bootstrap
    (FAB) model. """
    def __init__(self,
                 flow: TrainableDistribution,
                 target_distribution: TargetDistribution,
                 n_intermediate_distributions: int,
                 alpha: float = 2.,
                 transition_operator: Optional[TransitionOperator] = None,
                 ais_distribution_spacing: "str" = "linear",
                 loss_type: Optional["str"] = None,
                 use_ais: bool = True,
                 ):
        """
        Args:
            flow: Trainable flow model.
            target_distribution: Target distribution to fit.
            n_intermediate_distributions: Number of intermediate AIS distributions.
            alpha: Value of alpha if using fab_alpha_div loss.
            transition_operator: Transition operator for AIS.
            ais_distribution_spacing: AIS spacing type "geometric" or "linear"
            loss_type: Loss type for training. May be set to None if `self.loss` is not used.
                E.g. for training with the prioritised buffer.
            use_ais: Whether or not to use AIS. For losses that do not rely on AIS, this may still
                be set to True if we wish to use AIS in evaluation, which is why it is set to True
                by default.
        """
        assert loss_type in [None, "fab_ub_alpha_2_div",
                             "forward_kl", "flow_alpha_2_div",
                             "flow_reverse_kl", "fab_alpha_div",
                             "flow_alpha_2_div_unbiased", "flow_alpha_2_div_nis",
                             "target_forward_kl"]
        if loss_type in EXPERIMENTAL_LOSSES:
            raise Exception("Running using experiment loss not used within the main FAB paper.")
        if loss_type in ALPHA_DIV_TARGET_LOSSES:
            assert alpha is not None, "Alpha must be specified if using the alpha div loss."
        self.alpha = alpha
        self.loss_type = loss_type
        self.flow = flow
        self.target_distribution = target_distribution
        self.n_intermediate_distributions = n_intermediate_distributions
        self.ais_distribution_spacing = ais_distribution_spacing
        assert len(flow.event_shape) == 1, "Currently only 1D distributions are supported"
        if use_ais or loss_type in LOSSES_USING_AIS:
            if transition_operator is None:
                raise Exception("If using AIS, transition operator must be provided.")
            self.transition_operator = transition_operator
            self.annealed_importance_sampler = AnnealedImportanceSampler(
                base_distribution=self.flow,
                target_log_prob=self.target_distribution.log_prob,
                transition_operator=self.transition_operator,
                n_intermediate_distributions=self.n_intermediate_distributions,
                distribution_spacing_type=self.ais_distribution_spacing,
                p_target=False,
                alpha=self.alpha
            )

    def parameters(self):
        return self.flow.parameters()

    def loss(self, args) -> torch.Tensor:
        if self.loss_type is None:
            raise NotImplementedError("If loss_type is None, then the loss must be "
                                      "manually calculated.")
        if self.loss_type == "fab_alpha_div":
            # FAB loss estimated with AIS targeting minimum var IS distribution.
            return self.fab_alpha_div(args)
        elif self.loss_type == "forward_kl":
            return self.forward_kl(args)
        elif self.loss_type == "flow_reverse_kl":
            return self.flow_reverse_kl(args)
        elif self.loss_type == "flow_alpha_2_div":
            return self.flow_alpha_2_div(args)
        elif self.loss_type == "flow_alpha_2_div_unbiased":
            return self.flow_alpha_2_div_unbiased(args)
        elif self.loss_type == "flow_alpha_2_div_nis":
            return self.flow_alpha_2_div_nis(args)
        elif self.loss_type == "target_forward_kl":
            return self.target_forward_kl(args)
        elif self.loss_type == "fab_ub_alpha_2_div":
            return self.fab_ub_alpha_div_loss(args)
        else:
            raise NotImplementedError

    def set_ais_target(self, min_is_target: bool = True):
        """Set target to minimum importance sampling distribution for estimating the loss.
        if False, then the AIS target is set to p."""
        if not min_is_target:
            self.annealed_importance_sampler.p_target = True
            self.annealed_importance_sampler.transition_operator.p_target = True
        else:
            self.annealed_importance_sampler.p_target = False
            self.annealed_importance_sampler.transition_operator.p_target = False

    def fab_alpha_div_inner(self, point: Point, log_w_ais: torch.Tensor) -> \
            torch.Tensor:
        """Compute FAB loss based off points and importance weights from AIS targetting
        p^\alpha/q^{\alpha-1}.
        """
        log_q_x = self.flow.log_prob(point.x)
        return - np.sign(self.alpha) * torch.mean(torch.softmax(log_w_ais, dim=-1) * log_q_x)

    def fab_alpha_div(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss with p^\alpha/q^{\alpha-1} as the AIS target."""
        self.set_ais_target(min_is_target=True)
        point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        loss = self.fab_alpha_div_inner(point_ais, log_w_ais)
        # Reset ais target distribution back to p, which ensures evaluation is performed
        # with the target distribution.
        self.set_ais_target(min_is_target=False)
        return loss

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

    def inner_loss(self, point: Point, log_w_ais) -> torch.Tensor:
        """Loss as a function of AIS points and weights."""
        if self.loss_type == "fab_alpha_div":
            return self.fab_alpha_div_inner(point, log_w_ais)
        elif self.loss_type == "fab_ub_alpha_2_div":
            return self.fab_ub_alpha_div_loss_inner(point, log_w_ais)
        else:
            raise NotImplementedError

    def fab_ub_alpha_div_loss_inner(self, point: Point, log_w_ais: torch.Tensor) -> torch.Tensor:
        """Compute the FAB loss based on upper-bound of alpha-divergence with alpha=2 from
        https://arxiv.org/abs/2111.11510."""
        log_q_x = self.flow.log_prob(point.x)
        log_w = point.log_p - log_q_x
        return torch.logsumexp(log_w_ais + log_w, dim=0)

    def fab_ub_alpha_div_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        point_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        loss = self.fab_ub_alpha_div_loss_inner(point_ais.log_q, point_ais.log_p, log_w_ais)
        return loss

    def target_forward_kl(self, batch_size: int) -> torch.Tensor:
        """Assumes we can sample from the target distribution"""
        x = self.target_distribution.sample((batch_size,))
        return self.forward_kl(x)

    def forward_kl(self, x_p: torch.Tensor) -> torch.Tensor:
        """Forward kl with estimated using x ~ p(x) where p is the target distribution."""
        return -torch.mean(self.flow.log_prob(x_p))

    def get_iter_info(self) -> Dict[str, Any]:
        if hasattr(self, "annealed_importance_sampler"):
            if hasattr(self.annealed_importance_sampler, "_logging_info"):
                return self.annealed_importance_sampler.get_logging_info()
        return {}

    def get_eval_info(self,
                      outer_batch_size: int,
                      inner_batch_size: int,
                      set_p_target: bool = True,
                      ais_only: bool = False
                      ) -> Dict[str, Any]:
        if hasattr(self, "annealed_importance_sampler"):
            if set_p_target:
                self.set_ais_target(min_is_target=False)  # Evaluate with target=p.
            base_samples, base_log_w, ais_samples, ais_log_w = \
                self.annealed_importance_sampler.generate_eval_data(outer_batch_size,
                                                                    inner_batch_size)
            info = {"eval_ess_flow": effective_sample_size(log_w=base_log_w, normalised=False).item(),
                    "eval_ess_ais": effective_sample_size(log_w=ais_log_w, normalised=False).item()}

            if not ais_only:
                flow_info = self.target_distribution.performance_metrics(base_samples, base_log_w,
                                                                         self.flow.log_prob,
                                                                         batch_size=inner_batch_size)
                info.update({"flow_" + key: val for key, val in flow_info.items()})
            ais_info = self.target_distribution.performance_metrics(ais_samples, ais_log_w)
            info.update({"ais_" + key: val for key, val in ais_info.items()})

            # Back to target = p^\alpha & q^(1-\alpha).
            self.set_ais_target(min_is_target=True)

        else:
            raise NotImplementedError
            # TODO
        return info

    def save(self,
             path: "str"
             ):
        """Save FAB model to file."""
        torch.save({'flow': self.flow.state_dict(),
                    'trans_op': self.transition_operator.state_dict()},
                   path)

    def load(self,
             path: "str",
             map_location: Optional[str] = None,
             ):
        """Load FAB model from file."""
        checkpoint = torch.load(path, map_location=map_location)
        try:
            self.flow.load_state_dict(checkpoint['flow'])
        except:
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
        if self.annealed_importance_sampler:
            self.annealed_importance_sampler = AnnealedImportanceSampler(
                base_distribution=self.flow,
                target_log_prob=self.target_distribution.log_prob,
                transition_operator=self.transition_operator,
                p_target=False,
                alpha=self.alpha,
                n_intermediate_distributions=self.n_intermediate_distributions,
                distribution_spacing_type=self.ais_distribution_spacing)
