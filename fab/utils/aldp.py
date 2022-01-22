import torch
import numpy as np

import mdtraj
from matplotlib import pyplot as plt



def evaluateAldp(model, test_data, n_samples=1000, n_batches=1000,
                 save_path=None, data_path='.'):
    """
    Evaluate model of the Boltzmann distribution of the Alanine
    Dipeptide
    :param model: Model to be evaluated
    :param test_data: Torch array with test data
    :param n_samples: Int, number of samples to draw per batch
    :param n_batches: Int, number of batches to sample
    :param save_path: String, path where to save plots of marginals,
    if none plots are not created
    :param data_path: String, path to data used for transform init
    :return: KL divergences
    """
    # Set params for transform
    ndim = 66
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [6, 8, 9, 10, 14]

    # Load data for transform
    # Load the alanine dipeptide trajectory
    traj = mdtraj.load(data_path)
    traj.center_coordinates()

    # superpose on the backbone
    ind = traj.top.select("backbone")

    traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

    # Gather the training data into a pytorch Tensor with the right shape
    training_data = traj.xyz
    n_atoms = training_data.shape[1]
    n_dim = n_atoms * 3
    training_data_npy = training_data.reshape(-1, n_dim)
    training_data = torch.from_numpy(training_data_npy.astype("float64"))

    # Set up transform
    transform = bg.flows.CoordinateTransform(training_data, ndim,
                                             z_matrix, cart_indices)

    # Get test data
    z_d_np = test_data.cpu().data.numpy()
    x_d_np = np.zeros((0, 66))

    # Determine likelihood of test data
    log_p_sum = 0
    model_device = model.q0.loc.device
    for i in range(int(np.floor((len(test_data) - 1) / n_samples))):
        z = test_data[(i * n_samples):((i + 1) * n_samples), :]
        x, log_det = transform(z.cpu().double())
        x_d_np = np.concatenate((x_d_np, x.data.numpy()))
        log_p = model.log_prob(z.to(model_device))
        log_p_sum = log_p_sum + torch.sum(log_p).detach() - torch.sum(log_det).detach().float()
    z = test_data[((i + 1) * n_samples):, :]
    x, log_det = transform(z.cpu().double())
    x_d_np = np.concatenate((x_d_np, x.data.numpy()))
    log_p = model.log_prob(z.to(model_device))
    log_p_sum = log_p_sum + torch.sum(log_p).detach() - torch.sum(log_det).detach().float()
    log_p_avg = log_p_sum.cpu().data.numpy() / len(test_data)

    # Draw samples

    z_np = np.zeros((0, 60))
    x_np = np.zeros((0, 66))

    for i in range(n_batches):
        z, _ = model.sample(n_samples)
        x, _ = transform(z.cpu().double())
        x_np = np.concatenate((x_np, x.data.numpy()))
        z, _ = transform.inverse(x)
        z_np = np.concatenate((z_np, z.data.numpy()))

    # Estimate density of marginals
    nbins = 200
    hist_range = [-5, 5]
    ndims = z_np.shape[1]

    hists_test = np.zeros((nbins, ndims))
    hists_gen = np.zeros((nbins, ndims))

    for i in range(ndims):
        htest, _ = np.histogram(z_d_np[:, i], nbins, range=hist_range, density=True);
        hgen, _ = np.histogram(z_np[:, i], nbins, range=hist_range, density=True);
        hists_test[:, i] = htest
        hists_gen[:, i] = hgen

    # Compute KLD of marginals
    eps = 1e-10
    kld_unscaled = np.sum(hists_test * np.log((hists_test + eps) / (hists_gen + eps)), axis=0)
    kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins

    # Split KLD into groups
    ncarts = transform.mixed_transform.len_cart_inds
    permute_inv = transform.mixed_transform.permute_inv
    bond_ind = transform.mixed_transform.ic_transform.bond_indices
    angle_ind = transform.mixed_transform.ic_transform.angle_indices
    dih_ind = transform.mixed_transform.ic_transform.dih_indices

    kld_cart = kld[:(3 * ncarts - 6)]
    kld_ = np.concatenate([kld[:(3 * ncarts - 6)], np.zeros(6), kld[(3 * ncarts - 6):]])
    kld_ = kld_[permute_inv]
    kld_bond = kld_[bond_ind]
    kld_angle = kld_[angle_ind]
    kld_dih = kld_[dih_ind]

    # Compute Ramachandran plot angles
    test_traj = mdtraj.Trajectory(x_d_np.reshape(-1, 22, 3), traj.top)
    sampled_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), traj.top)
    psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
    psi_d[np.isnan(psi_d)] = 0
    phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
    phi_d[np.isnan(phi_d)] = 0
    psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
    psi[np.isnan(psi)] = 0
    phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)
    phi[np.isnan(phi)] = 0

    # Compute KLD of Ramachandran plot angles
    nbins_ram = 64
    eps_ram = 1e-10
    hist_ram_test = np.histogram2d(phi_d, psi_d, nbins_ram,
                                   range=[[-np.pi, np.pi], [-np.pi, np.pi]])[0]
    hist_ram_gen = np.histogram2d(phi, psi, nbins_ram,
                                  range=[[-np.pi, np.pi], [-np.pi, np.pi]])[0]
    kld_ram = np.mean(hist_ram_test / len(phi) * np.log(hist_ram_test + eps_ram)
                      / np.log(hist_ram_gen + eps_ram))

    # Create plots
    if save_path is not None:
        # Histograms of the groups
        hists_test_cart = hists_test[:, :(3 * ncarts - 6)]
        hists_test_ = np.concatenate([hists_test[:, :(3 * ncarts - 6)], np.zeros((nbins, 6)),
                                      hists_test[:, (3 * ncarts - 6):]], axis=1)
        hists_test_ = hists_test_[:, permute_inv]
        hists_test_bond = hists_test_[:, bond_ind]
        hists_test_angle = hists_test_[:, angle_ind]
        hists_test_dih = hists_test_[:, dih_ind]

        hists_gen_cart = hists_gen[:, :(3 * ncarts - 6)]
        hists_gen_ = np.concatenate([hists_gen[:, :(3 * ncarts - 6)], np.zeros((nbins, 6)),
                                     hists_gen[:, (3 * ncarts - 6):]], axis=1)
        hists_gen_ = hists_gen_[:, permute_inv]
        hists_gen_bond = hists_gen_[:, bond_ind]
        hists_gen_angle = hists_gen_[:, angle_ind]
        hists_gen_dih = hists_gen_[:, dih_ind]

        label = ['cart', 'bond', 'angle', 'dih']
        hists_test_list = [hists_test_cart, hists_test_bond, hists_test_angle, hists_test_dih]
        hists_gen_list = [hists_gen_cart, hists_gen_bond, hists_gen_angle, hists_gen_dih]
        x = np.linspace(*hist_range, nbins)
        for i in range(4):
            if i == 0:
                fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            else:
                fig, ax = plt.subplots(6, 3, figsize=(10, 20))
                ax[5, 2].set_axis_off()
            for j in range(hists_test_list[i].shape[1]):
                ax[j // 3, j % 3].plot(x, hists_test_list[i][:, j])
                ax[j // 3, j % 3].plot(x, hists_gen_list[i][:, j])
            plt.savefig(save_path + '_marginals_' + label[i] + '.png', dpi=300)
            plt.close()

        # Ramachandran plot
        plt.figure(figsize=(10, 10))
        plt.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm())
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\phi$', fontsize=24)
        plt.ylabel('$\psi$', fontsize=24)
        plt.savefig(save_path + '_ramachandran.png', dpi=300)
        plt.close()

    # Remove variables
    del x, z, transform, training_data

    return ((kld_cart, kld_bond, kld_angle, kld_dih), kld_ram, log_p_avg)