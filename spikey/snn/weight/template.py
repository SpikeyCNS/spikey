"""
The data structure to generate and manage connections between neurons.
Contains generation, arithmetic and get operations.
Updates are handled in spikey.snn.Synapse objects.
"""
import math
import numpy as np
from spikey.module import Module, Key


class Weight(Module):
    """
    The data structure to generate and manage connections between neurons.
    Contains generation, arithmetic and get operations.
    Updates are handled in spikey.snn.Synapse objects.

    .. note::
        Weight._matrix must be a masked ndarray with fill_value=0 while Weight.matrix
        is a simple ndarray.

        Arithmetic operations(a * b) use unmasked matrix for speed while inplace(a += b)
        arithmetic uses masked values.

        Get operations(Weight[[1, 2, 3]]) apply to masked ndarray.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        config = {
            "n_inputs": 1,
            "n_neurons": 10,
            "max_weight": 3,
        }
        w = Weight(**config)

        in_volts = w * np.ones(config['n_neurons'])

    .. code-block:: python

        class network_template(Network):
            keys = {
                "n_inputs": 1,
                "n_neurons": 10,
                "max_weight": 3,
            }
            parts = {
                "weights": Weight
            }
    """

    NECESSARY_KEYS = [
        Key("n_inputs", "Number of inputs.", int),
        Key("n_neurons", "Number of neurons in network.", int),
        Key("max_weight", "Max synapse weight.", float),
    ]

    def __init__(self, **kwargs):
        self._matrix = None
        super().__init__(**kwargs)

    def _assert_matrix_shape(self, matrix, key):
        expected_shape = (self._n_inputs + self._n_neurons, self._n_neurons)
        real_shape = matrix.shape
        if not np.array_equal(real_shape, expected_shape):
            base_error = f"Expected '{key}' shape to equal (N_INPUTS+N_NEURONS, N_NEURONS)[{expected_shape}], not {real_shape}!"
            if len(real_shape) > 2:
                raise ValueError(
                    base_error
                    + f" Squeeze extra single valued dimensions with `{key}.squeeze()`."
                )
            elif np.array_equal(real_shape, (self._n_neurons, self._n_neurons)):
                raise ValueError(
                    base_error + " Add N_INPUTS to the first dimension of your matrix."
                )
            elif np.array_equal(
                real_shape, (self._n_neurons, self._n_inputs + self._n_neurons)
            ):
                raise ValueError(base_error + f" Transpose your matrix with `{key}.T`.")
            else:
                raise ValueError(base_error)

    def _convert_feedforward(self, layers):
        """
        Convert network in feedforward layer format to weight matrix format.
        NOTE: Layers given as masked arrays will have masks dropped.

        Parameters
        ----------
        layers: [ndarray, ndarray, ...]
            Network to convert.

        Returns
        -------
        ndarray Network in weight matrix format.
        """
        matrix = np.zeros(
            (self._n_inputs + self._n_neurons, self._n_neurons), dtype=np.float
        )

        row_offset, col_offset = 0, 0
        for i, layer in enumerate(layers):
            n, m = layer.shape
            matrix[row_offset : row_offset + n, col_offset : col_offset + m] = layer
            row_offset += n
            col_offset += m

        return matrix

    @property
    def matrix(self) -> np.float:
        """
        Return unmasked weight matrix.
        """
        if isinstance(self._matrix, np.ma.core.MaskedArray):
            return self._matrix.data

        return self._matrix

    def clip(self):
        """
        Restrict weights to 0 and max_weight.
        """
        np.clip(self._matrix.data, 0.0, float(self._max_weight), out=self._matrix.data)

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

        self.clip()

        return self

    def __sub__(self, subtractor: np.ndarray) -> np.float:
        return self.matrix - subtractor

    def __isub__(self, subtractor: np.ndarray):
        self._matrix -= subtractor

        self.clip()

        return self

    def __mul__(self, multiplier: np.ndarray) -> np.float:
        return self.matrix * multiplier

    def __imul__(self, multiplier: np.ndarray):
        self._matrix *= multiplier

        self.clip()

        return self

    def __truediv__(self, divisor: np.ndarray) -> np.float:
        return self.matrix / divisor

    def __itruediv__(self, divisor: np.ndarray):
        self._matrix /= divisor

        self.clip()

        return self
