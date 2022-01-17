import abc
from typing import Any, Dict, List, Mapping, Union

import numpy as np

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    # copied from Acme: https://github.com/deepmind/acme
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""


class ListLogger(Logger):
    """Manually save the data to the class in a dict. Currently only supports scalar history
    inputs."""

    history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
    initialised = False

    def write(self, data: LoggingData) -> None:
        if not self.initialised:
            # first iteration we run checks and instantiate lists within for each key
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    assert np.size(value) == 1
                    value = float(value)
                else:
                    assert isinstance(value, float) or isinstance(value, int)
                self.history[key] = [value]
            self.initialised = True
        else:
            for key, value in data.items():
                value = float(value)
                self.history[key].append(value)

    def close(self) -> None:
        del self
