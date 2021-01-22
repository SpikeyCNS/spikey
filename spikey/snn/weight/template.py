"""
Template for weight matricies.

Override
"""
import math
import numpy as np


class Weight:
    """
    [n_inputs + n_neurons x n_neurons] Weight matrix.

    NOTE: matrix + x -- not masked values for speed reasons
          matrix += x is on masked values

    Parameters
    ----------
    n_inputs: int
        Number of inputs.
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    NECESSARY_KEYS = {
        "n_inputs": "int Number of inputs.",
        "n_neurons": "int Number of neurons in network.",
        "max_weight": "float Max synapse weight.",
    }

    def __init__(self, **kwargs):
        self._matrix = None

        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

    def _clip_weights(self):
        """
        Restrict weights to 0 and max weight.
        """
        self._matrix[~self._matrix.mask] = np.clip(
            self._matrix[~self._matrix.mask], 0, self._max_weight
        )

    @property
    def matrix(self) -> np.float:
        """
        Return the content of the weight matrix.
        """
        if isinstance(self._matrix, np.ma.core.MaskedArray):
            return self._matrix.data

        return self._matrix

    def __get__(self, obj: object, objtype: object) -> np.float:
        return self.matrix

    def __set__(self, obj: object, value: object):
        self.matrix = value

    def __getitem__(self, idx: np.int) -> np.float:
        return self._matrix[idx]

    def __add__(self, addend: np.ndarray) -> np.float:
        return self.matrix + addend

    def __iadd__(self, addend: np.ndarray):
        self._matrix += addend

        self._clip_weights()

        return self

    def __sub__(self, subtractor: np.ndarray) -> np.float:
        return self.matrix - subtractor

    def __isub__(self, subtractor: np.ndarray):
        self._matrix -= subtractor

        self._clip_weights()

        return self

    def __mul__(self, multiplier: np.ndarray) -> np.float:
        return self.matrix * multiplier

    def __imul__(self, multiplier: np.ndarray):
        self._matrix *= multiplier

        self._clip_weights()

        return self

    def __truediv__(self, divisor: np.ndarray) -> np.float:
        return self.matrix / divisor

    def __itruediv__(self, divisor: np.ndarray):
        self._matrix /= divisor

        self._clip_weights()

        return self
