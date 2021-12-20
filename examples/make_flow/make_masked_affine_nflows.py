from nflows import transforms, distributions, flows
from fab.wrappers import WrappedNFlowsModel
from fab.trainable_distributions import TrainableDistribution



def make_wrapped_nflows_dist(
        dim: int = 2,
        n_flow_layers: int = 5) -> TrainableDistribution:
    """Created a wrapped nflows distribution using the example from the nflows page."""

    # Define an invertible transformation.
    flow_layers = []
    for i in range(n_flow_layers):
        flow_layers.append(transforms.MaskedAffineAutoregressiveTransform(features=dim,
                                                                          hidden_features=4))
        flow_layers.append(transforms.RandomPermutation(features=dim))
        flow_layers.append(transforms.ActNorm(features=dim))
    transform = transforms.CompositeTransform(flow_layers)
    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[2])
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    wrapped_dist = WrappedNFlowsModel(flow)
    return wrapped_dist