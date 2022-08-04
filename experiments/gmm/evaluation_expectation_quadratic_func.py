import hydra

import pandas as pd
import os
from omegaconf import DictConfig
import torch
import numpy as np



from fab.target_distributions.gmm import GMM
from experiments.gmm.load_model_for_eval import load_model

PATH = os.getcwd()


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
    biases_unweighted = []
    for i in range(n_repeats):
        if model_name == "target":  # evaluate expectation using samples from the target.
            samples = target.sample((num_samples, ))
            log_w = torch.ones(samples.shape[0])
            samples_unweighted = samples  # fake
            log_w_unweighted = log_w  # fake
        else:
            if model_name and model_name[0:3] == "snf":
                use_snf = True
            else:
                use_snf = False
            path_to_model = f"{PATH}/models/{model_name}.pt"
            model = load_model(cfg, target, use_snf, path_to_model)
            if model_name and model_name[0:3] == "snf":
                samples, log_w = model.snf.sample(num_samples)
                valid_indices_unweighted = ~ (torch.softmax(log_w, axis=0) == 0)
                samples_unweighted = samples[valid_indices_unweighted]
                log_w_unweighted = torch.ones_like(log_w[valid_indices_unweighted])
            else:
                samples, log_q = model.flow.sample_and_log_prob((num_samples, ))
                log_w = target.log_prob(samples) - log_q
                valid_indices = ~torch.isinf(log_w) & ~torch.isnan(log_w)
                samples, log_w = samples[valid_indices], log_w[valid_indices]
                valid_indices_unweighted = ~ (torch.softmax(log_w, axis=0) == 0)
                samples_unweighted = samples[valid_indices_unweighted]
                log_w_unweighted = torch.ones_like(log_w[valid_indices_unweighted])
        normed_bias = target.evaluate_expectation(samples, log_w).detach().cpu()
        normed_bias_unweighted = target.evaluate_expectation(samples_unweighted,
                                                             log_w_unweighted).detach().cpu()
        biases.append(normed_bias)
        biases_unweighted.append(normed_bias_unweighted)
    info = {"bias": np.mean(np.abs(biases)),
            "std": np.std(biases),
            "bias_unweighted": np.abs(np.mean(biases_unweighted))}
    return info


@hydra.main(config_path="../config", config_name="gmm.yaml")
def main(cfg: DictConfig):
    model_names = ["target", "fab_buffer", "fab_no_buffer", "flow_kld", "flow_nis", "target_kld", "snf"]
    seeds = [1, 2, 3]
    num_samples = int(1000)

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

    fields = ["bias", "bias_unweighted"]
    print("\n *******  mean  ********************** \n")
    print(results.groupby("model_name").mean()[fields])
    print("\n ******* std ********************** \n")
    print(results.groupby("model_name").sem(ddof=0)[fields])
    results.to_csv(open(FILENAME_EXPECTATION_INFO, "w"))

FILENAME_EXPECTATION_INFO = "/examples/paper_results/gmm/gmm_results_expectation.csv"

if __name__ == '__main__':
    main()
