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
        config = {
            "n_inputs": 1,
            "n_neurons": 10,
            "max_weight": 3,
            "matrix": np.random.uniform(size=(1+10, 10)) <= .2,
        }
        _template_parts = {
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

        self._matrix = np.ma.copy(self._matrix)
        self._matrix = np.ma.clip(self._matrix, 0, self._max_weight)

        ## assert correct shape
        expected_shape = (self._n_inputs + self._n_neurons, self._n_neurons)
        real_shape = self.matrix.shape

        assert len(expected_shape) == len(
            real_shape
        ), f"Matrix shape not correct. Got: {real_shape}, Expected: {expected_shape}"
        for i in range(len(expected_shape)):
            assert (
                expected_shape[i] == real_shape[i]
            ), f"Matrix shape not correct. Got: {real_shape}, Expected: {expected_shape}"
