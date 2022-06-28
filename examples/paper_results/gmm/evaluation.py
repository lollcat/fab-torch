import hydra
from examples.setup_run_snf import make_normflow_snf_model, SNFModel

from fab.target_distributions.gmm import GMM
import pandas as pd
import os
from omegaconf import DictConfig
import torch

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



def evaluate(cfg: DictConfig, model_name: str, num_samples=int(1e4)):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.training.seed)
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling,
                 use_gpu=False, n_test_set_samples=num_samples)
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    model = load_model(cfg, target, model_name)
    eval = model.get_eval_info(num_samples, 500)
    return eval


@hydra.main(config_path="./", config_name="config.yaml")
def main(cfg: DictConfig):
    model_names = ["fab_buffer", "fab_no_buffer", "flow_kld", "flow_nis", "snf"]
    seeds = [1, 2, 3]
    num_samples = int(5e4)

    results = pd.DataFrame()
    for model_name in model_names:
        for seed in seeds:
            name = model_name + f"_seed{seed}"
            eval_info = evaluate(cfg, name, num_samples)
            eval_info.update(seed=seed,
                             model_name=model_name)
            results = results.append(eval_info, ignore_index=True)

    print("\n *******  mean  ********************** \n")
    print(results.groupby("model_name").mean()[["eval_ess_flow", "eval_ess_ais", "test_set_mean_log_prob"]])
    print("\n ******* std ********************** \n")
    print(results.groupby("model_name").std()[["eval_ess_flow", "eval_ess_ais", "test_set_mean_log_prob"]]*1.96)
    results.to_csv(open("/examples/paper_results/gmm/gmm_results.csv", "w"))
    print("overall results")
    print(results[["model_name", "seed", "eval_ess_flow", "eval_ess_ais", "test_set_mean_log_prob"]])


if __name__ == '__main__':
    if True:
        main()
    else:
        results = pd.read_csv(open("gmm_results.csv", "r"))
        print("mean")
        print(results.groupby("model_name").mean()[["eval_ess_flow", "eval_ess_ais", "test_set_mean_log_prob"]])
        print("std")
        print(results.groupby("model_name").std()[["eval_ess_flow", "eval_ess_ais", "test_set_mean_log_prob"]])
        results.to_csv(open("gmm_results.csv", "w"))
        print("overall results")
        print(results[["model_name", "seed", "eval_ess_flow", "eval_ess_ais", "test_set_mean_log_prob"]])
