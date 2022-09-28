import hydra

import pandas as pd
import os
from omegaconf import DictConfig
import torch

from fab.target_distributions.many_well import ManyWellEnergy
from experiments.load_model_for_eval import load_model


PATH = os.getcwd()


def evaluate_many_well(cfg: DictConfig, path_to_model: str, target, num_samples=int(5e4)):
    test_set_exact = target.sample((num_samples, ))
    test_set_log_prob_over_p = torch.mean(target.log_prob(test_set_exact) - target.log_Z).cpu().item()
    test_set_modes_log_prob_over_p = torch.mean(target.log_prob(target._test_set_modes) - target.log_Z)
    print(f"test set log prob under p: {test_set_log_prob_over_p:.2f}")
    print(f"modes test set log prob under p: {test_set_modes_log_prob_over_p:.2f}")
    model = load_model(cfg, target, path_to_model)
    eval = model.get_eval_info(num_samples, 500)
    return eval


@hydra.main(config_path="../config", config_name="many_well.yaml")
def main(cfg: DictConfig):
    """Evaluate each of the models, assume model checkpoints are saved as {model_name}_seed{i}.pt,
    where the model names for each method are `model_names` and `seeds` below."""
    # model_names = ["target_kld", "flow_nis", "flow_kld", "rbd", "snf_hmc", "fab_no_buffer",
    #                "fab_buffer"]
    model_names = ["rbd", "snf_hmc"]
    seeds = [1, 2, 3]
    num_samples = int(5e4)

    results = pd.DataFrame()
    for model_name in model_names:
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(cfg.training.seed)
        target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)
        if model_name and model_name[0:3] == "snf":
            # Update flow architecture for SNF if used.
            cfg.flow.use_snf = True
        else:
            cfg.flow.use_snf = False
        if model_name and model_name[0:3] == "rbd":
            cfg.flow.resampled_base = True
        else:
            cfg.flow.resampled_base = False
        if cfg.training.use_64_bit:
            torch.set_default_dtype(torch.float64)
            target = target.double()
        for seed in seeds:
            name = model_name + f"_seed{seed}"
            path_to_model = f"{PATH}/models/{name}.pt"
            print(f"get results for {name}")
            eval_info = evaluate_many_well(cfg, path_to_model, target, num_samples)
            eval_info.update(seed=seed,
                             model_name=model_name)
            results = results.append(eval_info, ignore_index=True)


    keys = ["eval_ess_flow", 'test_set_exact_mean_log_prob', 'test_set_modes_mean_log_prob',
            'MSE_log_Z_estimate', "forward_kl"]
    print("\n *******  mean  ********************** \n")
    print(results.groupby("model_name").mean()[keys].to_latex())
    print("\n ******* std ********************** \n")
    print((results.groupby("model_name").sem(ddof=0)[keys]).to_latex())
    results.to_csv(open(FILENAME_EVAL_INFO, "w"))

    print("overall results")
    print(results[["model_name", "seed"] + keys])

FILENAME_EVAL_INFO = "/experiments/many_well/many_well_results_iclr.csv"


if __name__ == '__main__':
    main()
