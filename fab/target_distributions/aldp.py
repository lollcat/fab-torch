import torch
from torch import nn

from fab.target_distributions.base import TargetDistribution

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from openmmtools.testsystems import AlanineDipeptideVacuum
import mdtraj
import tempfile



class AldpBoltzmann(nn.Module, TargetDistribution):
    def __init__(self, data_path=None, temperature=1000, energy_cut=1.e+8,
                 energy_max=1.e+20, n_threads=4, transform='mixed',
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.1, 'dih': 0.2}):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
        """
        super(AldpBoltzmann, self).__init__()

        # Define molecule parameters
        ndim = 66
        if transform == 'mixed':
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
        elif transform == 'internal':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
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
            cart_indices = [8, 6, 14]

        # System setup
        system = testsystems.AlanineDipeptideVacuum(constraints=None)
        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName('Reference'))

        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            testsystem = AlanineDipeptideVacuum(constraints=None)
            vacuum_sim = app.Simulation(testsystem.topology,
                                        testsystem.system,
                                        mm.LangevinIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond,
                                                              1.0 * unit.femtosecond),
                                        platform=mm.Platform.getPlatformByName('CPU'))
            vacuum_sim.context.setPositions(testsystem.positions)
            vacuum_sim.minimizeEnergy()
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + '/aldp.h5'
            vacuum_sim.reporters.append(mdtraj.reporters.HDF5Reporter(data_path, 10))
            vacuum_sim.step(10000)
            del (vacuum_sim)

        if data_path[-2:] == 'h5':
            # Load data for transform
            traj = mdtraj.load(data_path)
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path[-2:] == 'pt':
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError('Loading data or this format is not implemented.')

        # Set distribution
        self.coordinate_transform = bg.flows.CoordinateTransform(transform_data,
                                        ndim, z_matrix, cart_indices, mode=transform,
                                        ind_circ_dih=ind_circ_dih, shift_dih=shift_dih,
                                        shift_dih_params=shift_dih_params,
                                        default_std=default_std)

        if n_threads > 1:
            self.p = bg.distributions.TransformedBoltzmannParallel(system,
                            temperature, energy_cut=energy_cut, energy_max=energy_max,
                            transform=self.coordinate_transform, n_threads=n_threads)
        else:
            self.p = bg.distributions.TransformedBoltzmann(sim.context,
                            temperature, energy_cut=energy_cut, energy_max=energy_max,
                            transform=self.coordinate_transform)

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        return {}
