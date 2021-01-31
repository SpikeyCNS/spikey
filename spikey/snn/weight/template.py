"""
The data structure to generate and manage connections between neurons.
Contains generation, arithmetic and get operations.
Updates are handled in spikey.snn.Synapse objects.
"""
import math
import numpy as np


class Weight:
    """
    The data structure to generate and manage connections between neurons.
    Contains generation, arithmetic and get operations.
    Updates are handled in spikey.snn.Synapse objects.

    Notes
    -----
    - Weight._matrix must be a masked ndarray with fill_value=0 while Weight.matrix
    is a simple ndarray.
    - Arithmetic operations(a * b) use unmasked matrix for speed while inplace(a += b)
    arithmetic uses masked values.
    - Get operations(Weight[[1, 2, 3]]) apply to masked ndarray.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    config = {
        "n_inputs": 1,
        "n_neurons": 10,
        "max_weight": 3,
    }
    w = Weight(**config)

    in_volts = w * np.ones(config['n_neurons'])
    ```

    ```python
    class network_template(Network):
        config = {
            "n_inputs": 1,
            "n_neurons": 10,
            "max_weight": 3,
        }
        _template_parts = {
            "weights": Weight
        }
    ```
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
        Restrict weights to 0 and max_weight.
        """
        self._matrix[~self._matrix.mask] = np.clip(
            self._matrix[~self._matrix.mask], 0, self._max_weight
        )

    @property
    def matrix(self) -> np.float:
        """
        Return unmasked weight matrix.
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
