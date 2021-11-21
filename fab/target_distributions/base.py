import torch

class TargetDistribution(object):

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples, log_w):
        raise NotImplementedError
