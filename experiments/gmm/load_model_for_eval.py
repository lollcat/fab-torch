from typing import Optional
from experiments.setup_run_snf import make_normflow_snf_model, SNFModel
import os
from omegaconf import DictConfig
import torch

from fab import FABModel, HamiltonianMonteCarlo, Metropolis
from experiments.make_flow import make_wrapped_normflowdist

def load_model(cfg: DictConfig, target, use_snf: bool, path_to_model: Optional[str] = None):
    """Return the model with the loaded checkpoint provided in `path_to_model`.
    For evaluation we focus on evaluating the flow model (rather than AIS).
    if path is provided then the initialised model will be returned. cfg should be the config
    used across experiments (the specific loss etc of the cfg doesn't matter) as we are
    just evaluating the flow and not training. I.e. `config/gmm.yaml` for the GMM problem,
    and `config/many_well.yaml` for the Many Well problem."""
    dim = cfg.target.dim
    if use_snf:
        snf = make_normflow_snf_model(dim,
                                       n_flow_layers=cfg.flow.n_layers,
                                       layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                       act_norm=cfg.flow.act_norm,
                                       target=target
                                       )
        if path_to_model:
            checkpoint = torch.load(path_to_model, map_location="cpu")
            snf.load_state_dict(checkpoint['flow'])
        # wrap appropriately
        snf = SNFModel(snf, target, cfg.target.dim)
        return snf
    else:
        flow = make_wrapped_normflowdist(dim, n_flow_layers=cfg.flow.n_layers,
                                         layer_nodes_per_dim=cfg.flow.layer_nodes_per_dim,
                                         act_norm=cfg.flow.act_norm)
        if path_to_model:
            checkpoint = torch.load(path_to_model, map_location="cpu")
            flow._nf_model.load_state_dict(checkpoint['flow'])
        else:
            checkpoint = None

        if cfg.fab.transition_operator.type == "hmc":
            # very lightweight HMC.
            transition_operator = HamiltonianMonteCarlo(
                n_ais_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                n_outer=1,
                epsilon=1.0,
                L=cfg.fab.transition_operator.n_inner_steps,
                dim=dim,
                step_tuning_method="p_accept")
        elif cfg.fab.transition_operator.type == "metropolis":
            transition_operator = Metropolis(n_transitions=cfg.fab.n_intermediate_distributions,
                                             n_updates=cfg.fab.transition_operator.n_inner_steps,
                                             adjust_step_size=True)
        else:
            raise NotImplementedError

        if checkpoint:
            transition_operator.load_state_dict(checkpoint['trans_op'])

        model = FABModel(flow=flow,
                 target_distribution=target,
                 n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                 transition_operator=transition_operator,
                 loss_type=cfg.fab.loss_type)
    return model
