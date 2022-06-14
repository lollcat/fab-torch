from fab.target_distributions.many_well import ManyWellEnergy
import torch
import matplotlib.pyplot as plt

def test_many_well(dim: int = 6):
    assert dim % 2 == 0
    target = ManyWellEnergy(dim, a=-0.5, b=-6)
    sampler = torch.distributions.MultivariateNormal(loc=torch.ones(dim),
                                                              scale_tril=torch.eye(dim))
    samples = sampler.sample((100,))
    log_probs = target.log_prob(samples)

    assert torch.isfinite(log_probs).all()

    target.performance_metrics(samples=samples, log_w=torch.ones(samples.shape[0]),
                               log_q_fn=sampler.log_prob, batch_size=10)


    plt.plot(target.ais_x[:, 0], target.ais_x[:, 1], "o")
    plt.show()

    x = target.get_ais_based_test_set_samples(20)
    plt.plot(x[:, 2], x[:, 3], "o")
    plt.show()
    plt.plot(x[:, 6], x[:, 7], "o")
    plt.show()


    print(f"many well performance metrics over itself")
    print(target.performance_metrics(samples=samples, log_w=torch.ones(samples.shape[0]),
                               log_q_fn=target.log_prob, batch_size=500))

if __name__ == '__main__':
    test_many_well(32)
