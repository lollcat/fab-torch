import normflow as nf
from fab.wrappers.normflow import WrappedNormFlowModel
from fab.trainable_distributions import TrainableDistribution


def make_normflow_flow(dim: int,
                       n_flow_layers: int,
                       layer_nodes_per_dim: int):
    # Define list of flows
    flows = []
    layer_width = dim * layer_nodes_per_dim
    for i in range(n_flow_layers):
        # Neural network with two hidden layers having 32 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([int((dim / 2) + 0.5), layer_width, layer_width, dim], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="exp"))
        # Swap dimensions
        #flows.append(nf.flows.Permute(2, mode='swap'))
        # ActNorm
        #flows.append(nf.flows.ActNorm(dim))
    return flows


def make_wrapped_normflowdist(
        dim: int,
        n_flow_layers: int = 5,
        layer_nodes_per_dim: int = 10) -> TrainableDistribution:
    """Created a wrapped Normflow distribution using the example from the normflow page."""
    base = nf.distributions.base.DiagGaussian(dim)
    flows = make_normflow_flow(dim, n_flow_layers=n_flow_layers, layer_nodes_per_dim=layer_nodes_per_dim)
    model = nf.NormalizingFlow(base, flows)
    wrapped_dist = WrappedNormFlowModel(model)
    return wrapped_dist


def make_normflow_model(
        dim: int,
        target: nf.distributions.Target,
        n_flow_layers: int = 5) \
        -> nf.NormalizingFlow:
    """Created Normflow distribution using the example from the normflow page."""
    base = nf.distributions.base.DiagGaussian(dim)
    flows = make_normflow_flow(dim, n_flow_layers=n_flow_layers)
    model = nf.NormalizingFlow(base, flows, p=target)
    return model
