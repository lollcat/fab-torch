from typing import Optional, Tuple

import torch
from fab.trainable_distributions import TrainableDistribution




class DefensiveMixtureDistribution(TrainableDistribution):
    """Defensive mixture distribution.
    This does not have differerentiable sampling, so cannot be used for
    training by reverse KL minimisation. But may be used in AIS where we do not
    take the derivative."""
    def __init__(self, flow: TrainableDistribution,
                 defensive_dist: Optional[TrainableDistribution] = None):
        super(DefensiveMixtureDistribution, self).__init__()
        self.flow = flow
        assert len(self.flow.event_shape) == 1
        self.dim = self.flow.event_shape[0]

        flow_device = flow.sample((1,)).device
        if defensive_dist is None:
            self.loc = torch.nn.Parameter(torch.zeros(self.dim)).to(flow_device)
            self.log_scale = torch.nn.Parameter(torch.zeros(self.dim)).to(flow_device)
        else:
            self.defensive_dist = defensive_dist.to(flow_device)
        self.mixture_logit = torch.nn.Parameter(torch.tensor(1.0)).to(flow_device)
        assert self.defensive_dist.event_shape == (self.dim, )

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.dim, )

    @property
    def defensive_dist(self) -> torch.distributions.Distribution:
        scale = torch.exp(self.log_scale)
        return torch.distributions.Independent(torch.distributions.Normal(
            loc=self.loc, scale=scale, validate_args=False,
        ), reinterpreted_batch_ndims=1)
        # return torch.distributions.Independent(torch.distributions.Laplace(
        #     loc=self.loc, scale=scale, validate_args=False,
        # ), reinterpreted_batch_ndims=1)


    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_q_flow = self.flow.log_prob(x)
        mix_flow = torch.sigmoid(self.mixture_logit)
        log_q_defensive = self.defensive_dist.log_prob(x)

        components = torch.stack((
                log_q_flow + torch.log(mix_flow),
                log_q_defensive + torch.log(1 - mix_flow)
        ), dim=0)
        return torch.logsumexp(components, dim=0)

    @torch.no_grad()
    def sample(self, shape: Tuple) -> torch.Tensor:
        # NB: This is not differentiable.
        indices = torch.distributions.Binomial(
            logits=self.mixture_logit,
        ).sample(shape)
        samples_flow = self.flow.sample(shape)
        samples_defence = self.defensive_dist.sample(shape)
        samples = indices[..., None] * samples_flow + (1 - indices[..., None]) * samples_defence
        return samples

    def sample_and_log_prob(self, shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        # NB: this is not differentiable.
        samples = self.sample(shape)
        log_prob = self.log_prob(samples)
        return samples, log_prob


if __name__ == '__main__':
    from experiments.make_flow import make_wrapped_normflow_realnvp
    from fab.utils.plotting import plot_contours, plot_marginal_pair
    import matplotlib.pyplot as plt

    dim = 2
    flow = make_wrapped_normflow_realnvp(dim=dim,
                                         n_flow_layers=2,
                                         layer_nodes_per_dim=2)

    defensive_dist = DefensiveMixtureDistribution(flow=flow)

    batch_size = 128
    samples_flow = flow.sample((batch_size,))
    samples_df = defensive_dist.sample((batch_size,))

    fig, axs = plt.subplots(1, 2)
    plot_marginal_pair(samples_flow, ax=axs[0])
    plot_marginal_pair(samples_df, ax=axs[1])
    plot_contours(flow.log_prob, ax=axs[0])
    plot_contours(defensive_dist.log_prob, ax=axs[1])
    axs[0].set_title("flow dist samples and contours")
    axs[1].set_title("defensive dist samples and contours")
    plt.show()



