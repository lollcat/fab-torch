import torch
import numpy as np

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import AlanineDipeptideVacuum
import mdtraj
import tempfile

from fab.target_distributions.aldp import AldpBoltzmann



def test_aldp():
    # Generate a trajectory for coordinate transform
    temperature = 1000
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
    del(vacuum_sim)

    # Create distribution with data_path specified
    aldp_boltz = AldpBoltzmann(data_path)

    # Load test data
    traj = mdtraj.load(data_path)
    traj.center_coordinates()

    # superpose on the backbone
    ind = traj.top.select("backbone")
    traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

    # Gather the training data into a pytorch Tensor with the right shape
    test_data = traj.xyz
    n_atoms = test_data.shape[1]
    n_dim = n_atoms * 3
    test_data_npy = test_data.reshape(-1, n_dim)
    test_data = torch.from_numpy(test_data_npy.astype("float64"))

    # Transform coordinates
    x_test, log_det = aldp_boltz.coordinate_transform.inverse(test_data[::20])

    # Compute probability
    logp = aldp_boltz.log_prob(x_test)
    logp_np = logp.cpu().numpy()

    # Tests
    assert logp.shape == (50,)
    assert np.all(logp_np < -200)
    assert np.all(logp_np > -300)

    # Print sample values
    print("Log prob transformed Boltzmann distribution")
    print(logp)
    print("Log prob Boltzmann distribution")
    print(logp + log_det)

    # Create distribution without data_path specified
    aldp_boltz = AldpBoltzmann()

    # Transform coordinates
    x_test, log_det = aldp_boltz.coordinate_transform.inverse(test_data[::20])

    # Compute probability
    logp = aldp_boltz.log_prob(x_test)
    logp_np = logp.cpu().numpy()

    # Tests
    assert logp.shape == (50,)
    assert np.all(logp_np < -200)
    assert np.all(logp_np > -400)

    # Print sample values
    print("Log prob transformed Boltzmann distribution")
    print(logp)
    print("Log prob Boltzmann distribution")
    print(logp + log_det)

    

