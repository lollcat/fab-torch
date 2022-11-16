import hydra

import pandas as pd
import os
from omegaconf import DictConfig
import torch
import numpy as np

from experiments.gmm.evaluation import setup_target
from experiments.load_model_for_eval import load_model

PATH = os.getcwd()


def evaluate(cfg: DictConfig, path_to_model: str, target, num_samples=int(1e3), n_repeats=100):
    """Evaluate by estimating quadratic function. If `path_to_model==target` then this is done
    using samples from the target"""
    biases = []
    biases_unweighted = []
    for i in range(n_repeats):
        if path_to_model == "target":  # evaluate expectation using samples from the target.
            samples = target.sample((num_samples, ))
            log_w = torch.ones(samples.shape[0])
            samples_unweighted = samples  # fake
            log_w_unweighted = log_w  # fake
        else:
            model = load_model(cfg, target, path_to_model)
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
            "bias_unweighted": np.mean(np.abs(biases_unweighted))}
    return info


@hydra.main(config_path="../config", config_name="gmm.yaml")
def main(cfg: DictConfig):
    # model_names = ["target_kld", "fab_no_buffer", "fab_buffer"]
    model_names = ["target_kld", "flow_nis", "flow_kld", "rsb", "snf",
                   "fab_no_buffer", "fab_buffer"]
    seeds = [0, 1, 2]
    num_samples = int(1000)
    target = setup_target(cfg, num_samples)

    results = pd.DataFrame()
    for model_name in model_names:
        if model_name and model_name[0:3] == "snf":
            # Update flow architecture for SNF if used.
            cfg.flow.use_snf = True
        else:
            cfg.flow.use_snf = False
        if model_name and model_name[0:3] == "rsb":
            cfg.flow.resampled_base = True
        else:
            cfg.flow.resampled_base = False
        if model_name == "target":
            path_to_model = "target"
        else:
            path_to_model = f"{PATH}/models/{model_name}.pt"
        print(model_name)
        for seed in seeds:
            name = model_name + f"_seed{seed}"
            eval_info = evaluate(cfg, path_to_model, target, num_samples)
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


# use base config of GMM but overwrite for specific model.
@hydra.main(config_path="../config", config_name="gmm.yaml")
def alpha_study(cfg: DictConfig):
    alpha_values = [0.25,  0.5, 1.0, 1.5, 2.0, 3.0]
    seeds = [0, 1, 2]
    num_samples = int(1000)
    target = setup_target(cfg, num_samples)
    results = pd.DataFrame()
    for fab_type in ["buff", "no_buff"]:
        for alpha in alpha_values:
            for seed in seeds:
                name_without_seed = f"{fab_type}_alpha{alpha}"
                name = name_without_seed + f"_seed{seed}"
                path_to_model = f"{PATH}/models_alpha/{name}.pt"
                eval_info = evaluate(cfg, path_to_model, target, num_samples)
                eval_info.update(seed=seed,
                                 model_name=name_without_seed)
                results = results.append(eval_info, ignore_index=True)
    fields = ["bias", "bias_unweighted"]
    print("\n *******  mean  ********************** \n")
    print(results.groupby("model_name").mean()[fields])
    print("\n ******* std ********************** \n")
    print(results.groupby("model_name").sem(ddof=0)[fields])
    results.to_csv(open(FILENAME_EXPECTATION_ALPHA_INFO, "w"))


FILENAME_EXPECTATION_INFO = PATH + "/gmm_results_expectation.csv"
FILENAME_EXPECTATION_ALPHA_INFO = PATH + "/gmm_results_alpha_expectation.csv"

if __name__ == '__main__':
    # main()
    alpha_study()
