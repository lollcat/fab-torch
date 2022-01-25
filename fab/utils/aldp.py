import os
import torch
import numpy as np

import mdtraj
import matplotlib as mpl
from matplotlib import pyplot as plt
from openmmtools.testsystems import AlanineDipeptideVacuum



def evaluateAldp(z_sample, z_test, log_prob, transform,
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
        z = z_test[(i * batch_size):end, :]
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
    ncarts = transform.mixed_transform.len_cart_inds
    permute_inv = transform.mixed_transform.permute_inv.cpu().data.numpy()
    bond_ind = transform.mixed_transform.ic_transform.bond_indices.cpu().data.numpy()
    angle_ind = transform.mixed_transform.ic_transform.angle_indices.cpu().data.numpy()
    dih_ind = transform.mixed_transform.ic_transform.dih_indices.cpu().data.numpy()

    kld_cart = kld[:(3 * ncarts - 6)]
    kld_ = np.concatenate([kld[:(3 * ncarts - 6)], np.zeros(6), kld[(3 * ncarts - 6):]])
    kld_ = kld_[permute_inv]
    kld_bond = kld_[bond_ind]
    kld_angle = kld_[angle_ind]
    kld_dih = kld_[dih_ind]

    # Compute Ramachandran plot angles
    aldp = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    test_traj = mdtraj.Trajectory(x_d_np.reshape(-1, 22, 3), topology)
    sampled_traj = mdtraj.Trajectory(x_np.reshape(-1, 22, 3), topology)
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

    # Save metrics
    if metric_dir is not None:
        # Calculate and save KLD stats of marginals
        kld = (kld_cart, kld_bond, kld_angle, kld_dih)
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
        kld_labels = ['cart', 'bond', 'angle', 'dih']
        for kld_label, kld_ in zip(kld_labels, kld):
            kld_append = np.concatenate([np.array([iter + 1, np.median(kld_), np.mean(kld_)]), kld_])
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
        kld_append = np.array([[iter + 1, kld_ram]])
        if os.path.exists(kld_path):
            kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=',')
            if len(kld_hist.shape) == 1:
                kld_hist = kld_hist[None, :]
            kld_hist = np.concatenate([kld_hist, kld_append])
        else:
            kld_hist = kld_append
        np.savetxt(kld_path, kld_hist, delimiter=',',
                   header='it,kld', comments='')

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
        hists_test_list = [hists_test_cart, hists_test_bond,
                           hists_test_angle, hists_test_dih]
        hists_gen_list = [hists_gen_cart, hists_gen_bond,
                          hists_gen_angle, hists_gen_dih]
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
            plt.savefig(os.path.join(plot_dir, 'marginals_' + label[i] + '_' + str(iter) + '.png'),
                        dpi=300)
            plt.close()

        # Ramachandran plot
        plt.figure(figsize=(10, 10))
        plt.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm())
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$\phi$', fontsize=24)
        plt.ylabel('$\psi$', fontsize=24)
        plt.savefig(os.path.join(plot_dir, 'ramachandran_' + str(iter) + '.png'),
                    dpi=300)
        plt.close()