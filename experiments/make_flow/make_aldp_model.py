import torch
import numpy as np

import normflows as nf
import larsflow as lf

from fab.target_distributions.aldp import AldpBoltzmann
from fab.sampling_methods.transition_operators import HamiltonianMonteCarlo, Metropolis
from fab.wrappers.normflows import WrappedNormFlowModel
from fab import FABModel
from fab.core import ALPHA_DIV_TARGET_LOSSES


def make_aldp_model(config, device):
    # Set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)

    # Set up model

    # Target distribution
    transform_mode = 'mixed' if not 'transform' in config['system'] \
        else config['system']['transform']
    shift_dih = False if not 'shift_dih' in config['system'] \
        else config['system']['shift_dih']
    env = 'vacuum' if not 'env' in config['system'] \
        else config['system']['env']
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    target = AldpBoltzmann(data_path=config['data']['transform'],
                           temperature=config['system']['temperature'],
                           energy_cut=config['system']['energy_cut'],
                           energy_max=config['system']['energy_max'],
                           n_threads=config['system']['n_threads'],
                           transform=transform_mode,
                           ind_circ_dih=ind_circ_dih,
                           shift_dih=shift_dih,
                           env=env)
    target = target.to(device)

    # Flow parameters
    flow_type = config['flow']['type']
    ndim = 60

    ncarts = target.coordinate_transform.transform.len_cart_inds
    permute_inv = target.coordinate_transform.transform.permute_inv.cpu().numpy()
    dih_ind_ = target.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
    std_dih = target.coordinate_transform.transform.ic_transform.std_dih.cpu()

    ind = np.arange(ndim)
    ind = np.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=np.int), ind[3 * ncarts - 6:]])
    ind = ind[permute_inv]
    dih_ind = ind[dih_ind_]

    ind_circ = dih_ind[ind_circ_dih]
    bound_circ = np.pi / std_dih[ind_circ_dih]

    tail_bound = 5. * torch.ones(ndim)
    tail_bound[ind_circ] = bound_circ

    circ_shift = None if not 'circ_shift' in config['flow'] \
        else config['flow']['circ_shift']

    # Base distribution
    if config['flow']['base']['type'] == 'gauss':
        base = nf.distributions.DiagGaussian(ndim,
                                             trainable=config['flow']['base']['learn_mean_var'])
    elif config['flow']['base']['type'] == 'gauss-uni':
        base_scale = torch.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2
        base = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
        base.shape = (ndim,)
    elif config['flow']['base']['type'] == 'resampled-gauss-uni':
        base_scale = torch.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2
        base_ = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
        pf = nf.utils.nn.PeriodicFeaturesCat(ndim, ind_circ, np.pi / bound_circ)
        resnet = nf.nets.ResidualNet(ndim + len(ind_circ), 1, 
                                     config['flow']['base']['params']['a_hidden_units'],
                                     num_blocks=config['flow']['base']['params']['a_n_blocks'], 
                                     preprocessing=pf)
        a = torch.nn.Sequential(resnet, torch.nn.Sigmoid())
        base = lf.distributions.ResampledDistribution(base_, a, 
                                                      config['flow']['base']['params']['T'], 
                                                      config['flow']['base']['params']['eps'])
        base.shape = (ndim,)
    else:
        raise NotImplementedError('The base distribution ' + config['flow']['base']['type']
                                  + ' is not implemented')

    # Flow layers
    layers = []
    n_layers = config['flow']['blocks']

    for i in range(n_layers):
        if flow_type == 'rnvp':
            # Coupling layer
            hl = config['flow']['hidden_layers'] * [config['flow']['hidden_units']]
            scale_map = config['flow']['scale_map']
            scale = scale_map is not None
            if scale_map == 'tanh':
                output_fn = 'tanh'
                scale_map = 'exp'
            else:
                output_fn = None
            param_map = nf.nets.MLP([(ndim + 1) // 2] + hl + [(ndim // 2) * (2 if scale else 1)],
                                    init_zeros=config['flow']['init_zeros'], output_fn=output_fn)
            layers.append(nf.flows.AffineCouplingBlock(param_map, scale=scale,
                                                       scale_map=scale_map))
        elif flow_type == 'circular-ar-nsf':
            bl = config['flow']['blocks_per_layer']
            hu = config['flow']['hidden_units']
            nb = config['flow']['num_bins']
            ii = config['flow']['init_identity']
            dropout = config['flow']['dropout']
            layers.append(nf.flows.CircularAutoregressiveRationalQuadraticSpline(ndim,
                                                                                 bl, hu, ind_circ,
                                                                                 tail_bound=tail_bound, num_bins=nb,
                                                                                 permute_mask=True,
                                                                                 init_identity=ii,
                                                                                 dropout_probability=dropout))
        elif flow_type == 'circular-coup-nsf':
            bl = config['flow']['blocks_per_layer']
            hu = config['flow']['hidden_units']
            nb = config['flow']['num_bins']
            ii = config['flow']['init_identity']
            dropout = config['flow']['dropout']
            if i % 2 == 0:
                mask = nf.utils.masks.create_random_binary_mask(ndim, seed=seed + i)
            else:
                mask = 1 - mask
            layers.append(nf.flows.CircularCoupledRationalQuadraticSpline(ndim,
                                                                          bl, hu, ind_circ, tail_bound=tail_bound,
                                                                          num_bins=nb, init_identity=ii,
                                                                          dropout_probability=dropout, mask=mask))
        else:
            raise NotImplementedError('The flow type ' + flow_type + ' is not implemented.')

        if config['flow']['mixing'] == 'affine':
            layers.append(nf.flows.InvertibleAffine(ndim, use_lu=True))
        elif config['flow']['mixing'] == 'permute':
            layers.append(nf.flows.Permute(ndim))

        if config['flow']['actnorm']:
            layers.append(nf.flows.ActNorm(ndim))

        if i % 2 == 1 and i != n_layers - 1:
            if circ_shift == 'constant':
                layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ,
                                                     shift=bound_circ))
            elif circ_shift == 'random':
                gen = torch.Generator().manual_seed(seed + i)
                shift_scale = torch.rand([], generator=gen) + 0.5
                layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ,
                                                     shift=shift_scale * bound_circ))

        # SNF
        if 'snf' in config['flow']:
            if (i + 1) % config['flow']['snf']['every_n'] == 0:
                prop_scale = config['flow']['snf']['proposal_std'] * np.ones(ndim)
                steps = config['flow']['snf']['steps']
                proposal = nf.distributions.DiagGaussianProposal((ndim,), prop_scale)
                lam = (i + 1) / n_layers
                dist = nf.distributions.LinearInterpolation(target, base, lam)
                layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

    # Map input to periodic interval
    layers.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))

    # normflows model
    flow = nf.NormalizingFlow(base, layers)
    wrapped_flow = WrappedNormFlowModel(flow).to(device)

    # Set target of AIS to p or p^2/q
    min_is_target = config['fab']['loss_type'] in ALPHA_DIV_TARGET_LOSSES
    if 'replay_buffer' in config['training']:
        min_is_target = min_is_target or config['training']['replay_buffer']['type'] == 'prioritised'
    alpha = None if not 'alpha' in config['fab'] else config['fab']['alpha']


    # Transition operator
    transition_type = config['fab']['transition_type']
    if transition_type == 'hmc':
        # very lightweight HMC.
        transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=config['fab']['n_int_dist'],
            dim=ndim,
            base_log_prob=flow.log_prob,
            target_log_prob=target.log_prob,
            p_target=not min_is_target,
            alpha=alpha,
            L=config['fab']['n_inner'],
            epsilon=config['fab']['epsilon'] / 2,
            common_epsilon_init_weight=config['fab']['epsilon'] / 2)
        if not config['fab']['adjust_step_size']:
            transition_operator.set_eval_mode(True)
    elif transition_type == 'metropolis':
        transition_operator = Metropolis(
            n_ais_intermediate_distributions=config['fab']['n_int_dist'],
            dim=ndim,
            base_log_prob=flow.log_prob,
            target_log_prob=target.log_prob,
            p_target=not min_is_target,
            alpha=alpha,
            n_updates=config['fab']['n_inner'],
            max_step_size=config['fab']['max_step_size'],
            min_step_size=config['fab']['min_step_size'],
            adjust_step_size=config['fab']['adjust_step_size'])
    else:
        raise NotImplementedError('The transition operator ' + config['fab']['transition_type']
                                  + ' is not implemented')
    transition_operator = transition_operator.to(device)

    # FAB model
    loss_type = config['fab']['loss_type'] if 'loss_type' in config['fab'] else None
    model = FABModel(flow=wrapped_flow,
                     target_distribution=target,
                     n_intermediate_distributions=config['fab']['n_int_dist'],
                     transition_operator=transition_operator,
                     loss_type=loss_type,
                     alpha=alpha)
    return model
