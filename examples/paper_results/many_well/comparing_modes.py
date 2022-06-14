import hydra
import torch
from omegaconf import DictConfig
from examples.make_flow import make_wrapped_normflowdist
from fab.target_distributions.many_well import ManyWellEnergy
import os


PATH = os.getcwd()

@hydra.main(config_path="./", config_name="config.yaml")
def run(cfg: DictConfig):
    dim = cfg.target.dim
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)
    test_set_target_log_prob = target.log_prob(target._test_set)
    biggest_mode = torch.argmax(test_set_target_log_prob)
    smallest_mode = torch.argmin(test_set_target_log_prob)
    flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                     layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                     act_norm=cfg.flow.act_norm)
    path_to_model = f"{PATH}/models/fab_with_buffer_model.pt"
    checkpoint = torch.load(path_to_model, map_location="cpu")
    flow._nf_model.load_state_dict(checkpoint['flow'])

    max_diff_targ = target.log_prob(target._test_set[biggest_mode][None, ...]) - target.log_prob(target._test_set[smallest_mode][None, ...])
    max_diff_flow = flow.log_prob(target._test_set[biggest_mode][None, ...]) - flow.log_prob(target._test_set[smallest_mode][None, ...])
    print(max_diff_flow, max_diff_targ)

    indxs = [-1, -432, -300, -1000]
    for compare_index in [-200, 0]:
        if compare_index == -200:
            print("comparing relative to a moderately big mode")
        else:
            print("comparing to the smallest mode")
        for indx_small in indxs:
            small_diff_targ = target.log_prob(target._test_set[compare_index][None, ...]) - target.log_prob(target._test_set[indx_small][None, ...])
            small_diff_flow = flow.log_prob(target._test_set[compare_index][None, ...]) - flow.log_prob(target._test_set[indx_small][None, ...])
            print(f"target diff: {small_diff_targ.item():.2f}, flow diff: {small_diff_flow.item():.2f}")



if __name__ == '__main__':
    run()