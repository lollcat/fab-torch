import os
import torch
import numpy as np

import boltzgen as bg
import mdtraj
import matplotlib as mpl
from matplotlib import pyplot as plt
from openmmtools.testsystems import AlanineDipeptideVacuum



def evaluate_aldp(z_sample, z_test, log_prob, transform,
                  iter, metric_dir=None, plot_dir=None,
                  batch_size=1000):
    """
    Evaluate model of the Boltzmann distribution of the
    Alanine Dipeptide
    :param z_sample: Samples from the model
    :param z_test: Test data
    :param log_prob: Function to evaluate the log
    probability
    :param transform: Coordinate transformation
    :param iter: Current iteration count used for
    labeling of the generated files
    :param metric_dir: Directory where to store
    evaluation metrics
    :param plot_dir: Directory where to store plots
    :param batch_size: Batch size when processing
    the data
    """
    # Get mode of transform
    if isinstance(transform.transform, bg.mixed.MixedTransform):
        transform_mode = 'mixed'
    elif isinstance(transform.transform, bg.internal.CompleteInternalCoordinateTransform):
        transform_mode = 'internal'
    else:
        raise NotImplementedError('The evaluation is not implemented '
                                  'for this transform.')
    # Determine likelihood of test data and transform it
    z_d_np = z_test.cpu().data.numpy()
    x_d_np = np.zeros((0, 66))
    log_p_sum = 0
    n_batches = int(np.ceil(len(z_test) / batch_size))
    for i in range(n_batches):
        if i == n_batches - 1:
            end = len(z_test)
        else:
            end = (i + 1) * batch_size
        z = z_test[(i * batch_size):end, :]
        x, log_det = transform(z.double())
        x_d_np = np.concatenate((x_d_np, x.cpu().data.numpy()))
        log_p = log_prob(z)
        log_p_sum = log_p_sum + torch.sum(log_p).detach() - torch.sum(log_det).detach().float()
    log_p_avg = log_p_sum.cpu().data.numpy() / len(z_test)

    # Transform samples
    z_np = np.zeros((0, 60))
    x_np = np.zeros((0, 66))
    n_batches = int(np.ceil(len(z_sample) / batch_size))
    for i in range(n_batches):
        if i == n_batches - 1:
            end = len(z_sample)
        else:
            end = (i + 1) * batch_size
        z = z_sample[(i * batch_size):end, :]
        x, _ = transform(z.double())
        x_np = np.concatenate((x_np, x.cpu().data.numpy()))
        z, _ = transform.inverse(x)
        z_np = np.concatenate((z_np, z.cpu().data.numpy()))


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
    ncarts = transform.transform.len_cart_inds
    permute_inv = transform.transform.permute_inv.cpu().data.numpy()
    bond_ind = transform.transform.ic_transform.bond_indices.cpu().data.numpy()
    angle_ind = transform.transform.ic_transform.angle_indices.cpu().data.numpy()
    dih_ind = transform.transform.ic_transform.dih_indices.cpu().data.numpy()

    kld_cart = kld[:(3 * ncarts - 6)]
    kld_ = np.concatenate([kld[:(3 * ncarts - 6)], np.zeros(6), kld[(3 * ncarts - 6):]])
    kld_ = kld_[permute_inv]
    kld_bond = kld_[bond_ind]
    kld_angle = kld_[angle_ind]
    kld_dih = kld_[dih_ind]
    if transform_mode == 'internal':
        kld_bond = np.concatenate((kld_cart[:2], kld_bond))
        kld_angle = np.concatenate((kld_cart[2:], kld_angle))

    # Compute Ramachandran plot angles
    aldp = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    test_traj = mdtraj.Trajectory(x_d_np.reshape(-1, 22, 3), topology)
    sampled_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), topology)
    psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
    phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
    not_nan = np.logical_not(is_nan)
    psi_d = psi_d[not_nan]
    phi_d = phi_d[not_nan]
    psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
    not_nan = np.logical_not(is_nan)
    psi = psi[not_nan]
    phi = phi[not_nan]

    # Compute KLD of phi and psi
    htest_phi, _ = np.histogram(phi_d, nbins, range=[-np.pi, np.pi], density=True);
    hgen_phi, _ = np.histogram(phi, nbins, range=[-np.pi, np.pi], density=True);
    kld_phi = np.sum(htest_phi * np.log((htest_phi + eps) / (hgen_phi + eps))) \
              * 2 * np.pi / nbins
    htest_psi, _ = np.histogram(psi_d, nbins, range=[-np.pi, np.pi], density=True);
    hgen_psi, _ = np.histogram(psi, nbins, range=[-np.pi, np.pi], density=True);
    kld_psi = np.sum(htest_psi * np.log((htest_psi + eps) / (hgen_psi + eps))) \
              * 2 * np.pi / nbins

    # Compute KLD of Ramachandran plot angles
    nbins_ram = 64
    eps_ram = 1e-10
    hist_ram_test = np.histogram2d(phi_d, psi_d, nbins_ram,
                                   range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                   density=True)[0]
    hist_ram_gen = np.histogram2d(phi, psi, nbins_ram,
                                  range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                  density=True)[0]
    kld_ram = np.sum(hist_ram_test * np.log((hist_ram_test + eps_ram)
                                            / (hist_ram_gen + eps_ram))) \
              * (2 * np.pi / nbins_ram) ** 2

    # Save metrics
    if metric_dir is not None:
        # Calculate and save KLD stats of marginals
        kld = (kld_bond, kld_angle, kld_dih)
        kld_labels = ['bond', 'angle', 'dih']
        if transform_mode == 'mixed':
            kld = (kld_cart,) + kld
            kld_labels = ['cart'] + kld_labels
        kld_ = np.concatenate(kld)
        kld_append = np.array([[iter + 1, np.median(kld_), np.mean(kld_)]])
        kld_path = os.path.join(metric_dir, 'kld.csv')
        if os.path.exists(kld_path):
            kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
            if len(kld_hist.shape) == 1:
                kld_hist = kld_hist[None, :]
            kld_hist = np.concatenate([kld_hist, kld_append])
        else:
            kld_hist = kld_append
        np.savetxt(kld_path, kld_hist, delimiter=',',
                   header='it,kld_median,kld_mean', comments='')
        for kld_label, kld_ in zip(kld_labels, kld):
            kld_append = np.concatenate([np.array([iter + 1, np.median(kld_), np.mean(kld_)]), kld_])
            kld_append = kld_append[None, :]
            kld_path = os.path.join(metric_dir, 'kld_' + kld_label + '.csv')
            if os.path.exists(kld_path):
                kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
                if len(kld_hist.shape) == 1:
                    kld_hist = kld_hist[None, :]
                kld_hist = np.concatenate([kld_hist, kld_append])
            else:
                kld_hist = kld_append
            header = 'it,kld_median,kld_mean'
            for kld_ind in range(len(kld_)):
                header += ',kld' + str(kld_ind)
            np.savetxt(kld_path, kld_hist, delimiter=',',
                       header=header, comments='')

        # Save KLD of Ramachandran and log_p
        kld_path = os.path.join(metric_dir, 'kld_ram.csv')
        kld_append = np.array([[iter + 1, kld_phi, kld_psi, kld_ram]])
        if os.path.exists(kld_path):
            kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
            if len(kld_hist.shape) == 1:
                kld_hist = kld_hist[None, :]
            kld_hist = np.concatenate([kld_hist, kld_append])
        else:
            kld_hist = kld_append
        np.savetxt(kld_path, kld_hist, delimiter=',',
                   header='it,kld_phi,kld_psi,kld_ram', comments='')

        # Save log probability
        log_p_append = np.array([[iter + 1, log_p_avg]])
        log_p_path = os.path.join(metric_dir, 'log_p_test.csv')
        if os.path.exists(log_p_path):
            log_p_hist = np.loadtxt(log_p_path, skiprows=1, delimiter=',')
            if len(log_p_hist.shape) == 1:
                log_p_hist = log_p_hist[None, :]
            log_p_hist = np.concatenate([log_p_hist, log_p_append])
        else:
            log_p_hist = log_p_append
        np.savetxt(log_p_path, log_p_hist, delimiter=',',
                   header='it,log_p', comments='')

    # Create plots
    if plot_dir is not None:
        # Histograms of the groups
        hists_test_cart = hists_test[:, :(3 * ncarts - 6)]
        hists_test_ = np.concatenate([hists_test[:, :(3 * ncarts - 6)],
                                      np.zeros((nbins, 6)),
                                      hists_test[:, (3 * ncarts - 6):]], axis=1)
        hists_test_ = hists_test_[:, permute_inv]
        hists_test_bond = hists_test_[:, bond_ind]
        hists_test_angle = hists_test_[:, angle_ind]
        hists_test_dih = hists_test_[:, dih_ind]

        hists_gen_cart = hists_gen[:, :(3 * ncarts - 6)]
        hists_gen_ = np.concatenate([hists_gen[:, :(3 * ncarts - 6)],
                                     np.zeros((nbins, 6)),
                                     hists_gen[:, (3 * ncarts - 6):]], axis=1)
        hists_gen_ = hists_gen_[:, permute_inv]
        hists_gen_bond = hists_gen_[:, bond_ind]
        hists_gen_angle = hists_gen_[:, angle_ind]
        hists_gen_dih = hists_gen_[:, dih_ind]

        if transform_mode == 'internal':
            hists_test_bond = np.concatenate((hists_test_cart[:, :2],
                                              hists_test_bond), 1)
            hists_gen_bond = np.concatenate((hists_gen_cart[:, :2],
                                             hists_gen_bond), 1)
            hists_test_angle = np.concatenate((hists_test_cart[:, 2:],
                                               hists_test_angle), 1)
            hists_gen_angle = np.concatenate((hists_gen_cart[:, 2:],
                                              hists_gen_angle), 1)

        label = ['bond', 'angle', 'dih']
        hists_test_list = [hists_test_bond, hists_test_angle,
                           hists_test_dih]
        hists_gen_list = [hists_gen_bond, hists_gen_angle,
                          hists_gen_dih]
        if transform_mode == 'mixed':
            label = ['cart'] + label
            hists_test_list = [hists_test_cart] + hists_test_list
            hists_gen_list = [hists_gen_cart] + hists_gen_list
        x = np.linspace(*hist_range, nbins)
        for i in range(len(label)):
            if transform_mode == 'mixed':
                ncol = 3
                if i == 0:
                    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
                else:
                    fig, ax = plt.subplots(6, 3, figsize=(10, 20))
                    ax[5, 2].set_axis_off()
            elif transform_mode == 'internal':
                ncol = 4
                if i == 0:
                    fig, ax = plt.subplots(6, 4, figsize=(15, 24))
                    for j in range(1, 4):
                        ax[5, j].set_axis_off()
                elif i == 2:
                    fig, ax = plt.subplots(5, 4, figsize=(15, 20))
                    ax[4, 3].set_axis_off()
                else:
                    fig, ax = plt.subplots(5, 4, figsize=(15, 20))
            for j in range(hists_test_list[i].shape[1]):
                ax[j // ncol, j % ncol].plot(x, hists_test_list[i][:, j])
                ax[j // ncol, j % ncol].plot(x, hists_gen_list[i][:, j])
            plt.savefig(os.path.join(plot_dir, 'marginals_%s_%07i.png' % (label[i], iter + 1)),
                        dpi=300)
            plt.close()

        # Plot phi and psi
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        x = np.linspace(-np.pi, np.pi, nbins)
        ax[0].plot(x, htest_phi, linewidth=3)
        ax[0].plot(x, hgen_phi, linewidth=3)
        ax[0].tick_params(axis='both', labelsize=20)
        ax[0].set_xlabel('$\phi$', fontsize=24)
        ax[1].plot(x, htest_psi, linewidth=3)
        ax[1].plot(x, hgen_psi, linewidth=3)
        ax[1].tick_params(axis='both', labelsize=20)
        ax[1].set_xlabel('$\psi$', fontsize=24)
        plt.savefig(os.path.join(plot_dir, 'phi_psi_%07i.png' % (iter + 1)),
                    dpi=300)
        plt.close()

        # Ramachandran plot
        plt.figure(figsize=(10, 10))
        plt.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm(),
                   range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\phi$', fontsize=24)
        plt.ylabel('$\psi$', fontsize=24)
        plt.savefig(os.path.join(plot_dir, 'ramachandran_%07i.png' % (iter + 1)),
                    dpi=300)
        plt.close()


def filter_chirality(x, ind=[17, 26], mean_diff=-0.043, threshold=0.8):
    """
    Filters batch for the L-form
    :param x: Input batch
    :param ind: Indices to be used for determining the chirality
    :param mean_diff: Mean of the difference of the coordinates
    :param threshold: Threshold to be used for splitting
    :return: Returns indices of batch, where L-form is present
    """
    diff_ = torch.column_stack((x[:, ind[0]] - x[:, ind[1]],
                                x[:, ind[0]] - x[:, ind[1]] + 2 * np.pi,
                                x[:, ind[0]] - x[:, ind[1]] - 2 * np.pi))
    min_diff_ind = torch.min(torch.abs(diff_), 1).indices
    diff = diff_[torch.arange(x.shape[0]), min_diff_ind]
    ind = torch.abs(diff - mean_diff) < threshold
    return ind