from typing import Optional
from omegaconf import DictConfig

from experiments.setup_run import setup_model


def load_model(cfg: DictConfig, target, path_to_model: Optional[str] = None,
               map_location: str = "cpu"):
    """Return the model with the loaded checkpoint provided in `path_to_model`."""
    model = setup_model(cfg, target)
    model.load(path_to_model, map_location=map_location)
    return model
