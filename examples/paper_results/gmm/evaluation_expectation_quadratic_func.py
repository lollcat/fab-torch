import hydra
from examples.setup_run_snf import make_normflow_snf_model, SNFModel

from fab.target_distributions.gmm import GMM
import pandas as pd
import os
from omegaconf import DictConfig
import torch
import numpy as np

from fab import FABModel, HamiltonianMonteCarlo, Metropolis
from examples.make_flow import make_wrapped_normflowdist


PATH = os.getcwd()

def load_model(cfg: DictConfig, target, model_name: str):
    dim = cfg.target.dim
    if model_name and model_name[0:3] == "snf":
        snf = make_normflow_snf_model(dim,
                                       n_flow_layers=cfg.flow.n_layers,
                                       layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                       act_norm=cfg.flow.act_norm,
                                       target=target
                                       )
        if model_name:
            path_to_model = f"{PATH}/models/{model_name}.pt"
            checkpoint = torch.load(path_to_model, map_location="cpu")
            snf.load_state_dict(checkpoint['flow'])
        # wrap appropriately
        snf = SNFModel(snf, target, cfg.target.dim)
        return snf
    else:
        flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                         layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                         act_norm=cfg.flow.act_norm)
        path_to_model = f"{PATH}/models/{model_name}.pt"
        checkpoint = torch.load(path_to_model, map_location="cpu")
        flow._nf_model.load_state_dict(checkpoint['flow'])

        transition_operator = Metropolis(n_transitions=cfg.fab.n_intermediate_distributions,
                                         n_updates=cfg.fab.transition_operator.n_inner_steps,
                                         adjust_step_size=True)
        model = FABModel(flow=flow,
                 target_distribution=target,
                 n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                 transition_operator=transition_operator,
                 loss_type=cfg.fab.loss_type)
    return model



def evaluate(cfg: DictConfig, model_name: str, num_samples=int(1e3), n_repeats=100):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.training.seed)
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling,
                 use_gpu=False, n_test_set_samples=num_samples)
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()

    biases = []
    for i in range(n_repeats):
        if model_name[:6] == "target":
            samples = target.sample((num_samples, ))
            log_w = torch.ones(samples.shape[0])
        else:
            model = load_model(cfg, target, model_name)
            if model_name and model_name[0:3] == "snf":
                samples, log_w = model.snf.sample(num_samples)
            else:
                samples, log_q = model.flow.sample_and_log_prob((num_samples, ))
                valid_indices = ~torch.isinf(log_q) & ~torch.isnan(log_q)
                samples, log_q = samples[valid_indices], log_q[valid_indices]
                log_w = target.log_prob(samples) - log_q
        normed_bias = target.evaluate_expectation(samples, log_w).detach().cpu()
        biases.append(normed_bias)
    info = {"bias": np.abs(np.mean(biases)),
            "std": np.std(biases)}
    return info


@hydra.main(config_path="./", config_name="config.yaml")
def main(cfg: DictConfig):
    model_names = ["target", "fab_buffer", "fab_no_buffer", "flow_kld", "flow_nis", "snf"]
    seeds = [1, 2, 3]
    num_samples = int(1e3)

    results = pd.DataFrame()
    for model_name in model_names:
        print(model_name)
        for seed in seeds:
            name = model_name + f"_seed{seed}"
            eval_info = evaluate(cfg, name, num_samples)
            eval_info.update(seed=seed,
                             model_name=model_name)
            results = results.append(eval_info, ignore_index=True)
        print(results)

    fields = ["bias", "std"]
    print("\n *******  mean  ********************** \n")
    print(results.groupby("model_name").mean()[fields])
    print("\n ******* std ********************** \n")
    print(results.groupby("model_name").std()[fields])
    results.to_csv(open(FILENAME, "w"))


if __name__ == '__main__':
    FILENAME = "/home/laurence/work/code/FAB-TORCH/examples/paper_results/gmm/gmm_results_expectation.csv"
    if True:
        main()
    else:
        results = pd.read_csv(open(FILENAME, "r"))
        fields = ["bias", "std"]
        print("\n *******  mean  ********************** \n")
        print(results.groupby("model_name").mean()[fields])
        print("\n ******* std ********************** \n")
        print(results.groupby("model_name").std()[fields])
