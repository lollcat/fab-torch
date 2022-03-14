import os

import yaml
import numpy as np
import torch


def load_config(path):
    """
    Read configuration parameter form file
    :param path: Path to the yaml configuration file
    :return: Dict with parameter
    """

    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


def get_latest_checkpoint(dir_path, key=''):
    """
    Get path to latest checkpoint in directory
    :param dir_path: Path to directory to search for checkpoints
    :param key: Key which has to be in checkpoint name
    :return: Path to latest checkpoint
    """
    if not os.path.exists(dir_path):
        return None
    checkpoints = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, f)) and key in f and ".pt" in f]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return checkpoints[-1]


class DatasetIterator:
    """Create an iterator that returns batches of data. This is useful for iterating through
    a dataset and performing multiple forward passes without overloading the GPU."""
    def __init__(self, batch_size: int, dataset: torch.Tensor, device):
        self.batch_size = batch_size
        self.n_splits = int(np.ceil(dataset.shape[0] / batch_size))  # roundup
        self.dataset_iter = iter(torch.split(dataset, self.n_splits))
        self.device = device
        self.test_set_n_points = dataset.shape[0]

    def __next__(self):
        return next(self.dataset_iter).to(self.device)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_splits