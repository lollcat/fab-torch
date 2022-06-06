from typing import Optional, Dict, Any
import torch

from fab.types_ import Model
from fab.target_distributions.base import TargetDistribution
from fab.trainable_distributions import TrainableDistribution
from fab.utils.numerical import effective_sample_size



class VanillaModel(Model):
    """A model that trains a flow through estimation of the loss with samples from the flow."""
    def __init__(self,
                 flow: TrainableDistribution,
                 target_distribution: TargetDistribution,
                 loss_type: "str" = "alpha_2_div",
                 ):
        assert loss_type in ["alpha_2_div", "reverse_kl"]
        self.loss_type = loss_type
        self.flow = flow
        self.target_distribution = target_distribution

    def parameters(self):
        return self.flow.parameters()

    def loss(self, args) -> torch.Tensor:
        if self.loss_type == "alpha_2_div":
            return self.alpha_div_loss(args)
        else:
            raise NotImplementedError

    def inner_loss(self, x_ais, log_w_ais) -> torch.Tensor:
        """Loss as a function of ais samples and weights, we use this when training with a replay buffer."""
        if self.loss_type == "alpha_2_div":
            return self.fab_alpha_div_loss_inner(x_ais, log_w_ais)
            raise NotImplementedError

    def fab_alpha_div_loss_inner(self, x_ais, log_w_ais) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        log_q_x = self.flow.log_prob(x_ais)
        log_p_x = self.target_distribution.log_prob(x_ais)
        log_w = log_p_x - log_q_x
        return torch.logsumexp(log_w_ais + log_w, dim=0)

    def fab_alpha_div_loss(self, batch_size: int) -> torch.Tensor:
        """Compute the FAB loss based on lower-bound of alpha-divergence with alpha=2."""
        x_ais, log_w_ais = self.annealed_importance_sampler.sample_and_log_weights(batch_size)
        x_ais = x_ais.detach()
        log_w_ais = log_w_ais.detach()
        loss = self.fab_alpha_div_loss_inner(x_ais, log_w_ais)
        return loss

    def get_iter_info(self) -> Dict[str, Any]:
        return {}

    def get_eval_info(self,
                      outer_batch_size: int,
                      inner_batch_size: int,
                      ) -> Dict[str, Any]:
        base_samples, base_log_q = self.flow.sample_and_log_prob((outer_batch_size,))
        base_log_w = self.target_distribution.log_prob(base_samples) - base_log_q
        info = {"eval_ess_flow": effective_sample_size(log_w=base_log_w, normalised=False).item()}
        flow_info = self.target_distribution.performance_metrics(base_samples, base_log_w,
                                                                 self.flow.log_prob,
                                                                 batch_size=inner_batch_size)
        info.update(flow_info)
        return info

    def save(self,
             path: "str"
             ):
        """Save FAB model to file."""
        torch.save({'flow': self.flow._nf_model.state_dict()})

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

