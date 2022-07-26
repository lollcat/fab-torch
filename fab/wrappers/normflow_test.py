from fab.wrappers.normflows import WrappedNormFlowModel
from fab.trainable_distributions import TrainableDistribution
import normflows as nf


def make_wrapped_normflowdist(
        dim: int = 2) -> TrainableDistribution:
    """Created a wrapped normflows distribution using the example from the normflows page."""
    base = nf.distributions.base.DiagGaussian(dim)

    # Define list of flows
    num_layers = 16
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 32 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 32, 32, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))

    model = nf.NormalizingFlow(base, flows)
    wrapped_dist = WrappedNormFlowModel(model)
    return wrapped_dist


def test_wrapped_normflowdist(
        dim: int = 2,
        batch_size: int = 10) -> None:
    wrapped_dist = make_wrapped_normflowdist(dim)
    samples, log_probs = wrapped_dist.sample_and_log_prob((batch_size,))
    assert samples.shape == (batch_size, dim)
    assert log_probs.shape == (batch_size,)
