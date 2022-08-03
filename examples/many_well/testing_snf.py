import hydra
from examples.setup_run_snf import make_normflow_snf_model, SNFModel

from fab.target_distributions.many_well import ManyWellEnergy
import pandas as pd
import os
from omegaconf import DictConfig
import torch
from copy import deepcopy

from fab import FABModel, HamiltonianMonteCarlo, Metropolis
from examples.make_flow import make_wrapped_normflowdist


PATH = os.getcwd()

def load_model(cfg: DictConfig, target, log_prob_scale_factor = 0.0):
    dim = cfg.target.dim
    target_snf = deepcopy(target)
    def target_log_prob(x):
        return target.log_prob(x) - log_prob_scale_factor
    print(log_prob_scale_factor)
    target_snf.log_prob = target_log_prob
    target_snf.scale_factor_log_prob = log_prob_scale_factor

    snf = make_normflow_snf_model(dim,
                                   n_flow_layers=cfg.flow.n_layers,
                                   layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                   act_norm=cfg.flow.act_norm,
                                   target=target_snf
                                   )
    # wrap appropriately
    snf = SNFModel(snf, target_snf, cfg.target.dim)
    return snf


def evaluate(cfg: DictConfig, target, num_samples=int(100), log_prob_scale_factor = 0.0):
    torch.manual_seed(0)
    test_set_ais = target.sample((num_samples,))
    model = load_model(cfg, target, log_prob_scale_factor=log_prob_scale_factor)
    torch.manual_seed(0)
    log_w_ais = model.snf.log_prob(test_set_ais)
    torch.manual_seed(0)
    x, log_w_self = model.snf.sample(num_samples)
    eval = {"log_w_ais": torch.mean(log_w_ais).cpu().detach().numpy(),
            "log_w_self": torch.mean(log_w_self).cpu().detach().numpy()
            }
    return eval


@hydra.main(config_path="../config", config_name="many_well.yaml")
def main(cfg: DictConfig):
    model_names = ["snf"]
    log_prob_scale_factors = [0, 1, 10, 10, 70]
    num_samples = int(1e3)

    results = pd.DataFrame()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.training.seed)
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)#,
                            # normalised=model_name == "snf")
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    for log_prob_scale_factor in log_prob_scale_factors:
        eval_info = evaluate(cfg, target, num_samples, log_prob_scale_factor=log_prob_scale_factor)
        eval_info.update(log_prob_scale_factor=log_prob_scale_factor)
        results = results.append(eval_info, ignore_index=True)
        print(f"run {log_prob_scale_factor}")
    print(results)
    pass

if __name__ == '__main__':
    main()
