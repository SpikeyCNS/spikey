"""
Manually defined network - directly by ndarray or with function.
The data structure to generate and manage connections between neurons.
Contains generation, arithmetic and get operations.
Updates are handled in spikey.snn.Synapse objects.
"""
import numpy as np
from spikey.module import Key
from spikey.snn.weight.template import Weight


class Manual(Weight):
    """
    Manually defined network - directly by ndarray or with function.
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
        "matrix": np.random.uniform(size=(1+10, 10)) <= .2,
    }
    w = Manual(**config)

    in_volts = w * np.ones(config['n_neurons'])
    ```

    ```python
    class network_template(Network):
        keys = {
            "n_inputs": 1,
            "n_neurons": 10,
            "max_weight": 3,
            "matrix": np.random.uniform(size=(1+10, 10)) <= .2,
        }
        parts = {
            "weights": Manual
        }
    ```
    """

    NECESSARY_KEYS = Weight.extend_keys(
        [
            Key("matrix", "ndarray/func Matrix to use/generate."),
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if callable(self._matrix):
            self._matrix = self._matrix(self)

        if not hasattr(self._matrix, "mask"):
            print(
                "WARNING: Converting weight Manual.matrix to masked array, masked on values == 0."
            )
            self._matrix = np.ma.array(
                self._matrix, mask=(self._matrix == 0), fill_value=0
            )
        else:
            self._matrix.fill_value = 0
        self._matrix = np.ma.copy(self._matrix)
        self._matrix = np.ma.clip(self._matrix, 0, self._max_weight)

        self._assert_matrix_shape(self._matrix, key="matrix")
