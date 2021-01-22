"""
Manual input of weight matrix.
"""
from copy import deepcopy

import numpy as np

from spikey.snn.weight.template import Weight


class Manual(Weight):
    """
    Manually create network. -- or generate based on function.

    Parameters
    ----------
    n_inputs: int
        Number of inputs.
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    NECESSARY_KEYS = deepcopy(Weight.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "matrix": "ndarray/func Matrix to use/generate.",
            "inh_weight_mask": "ndarray, boolean Matrix showing what synapses are inhibitory",
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if callable(self._matrix):
            self._matrix = self._matrix(self)

        self._matrix = np.ma.copy(self._matrix)

        self._matrix = np.ma.clip(self._matrix, 0, self._max_weight)

        if self._inh_weight_mask is not None:
            self.inh = np.where(self._inh_weight_mask, -1, 1)
        else:
            self.inh = np.ones(self._matrix.shape)

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

    def __mul__(self, multiplier):
        return self.matrix * self.inh * multiplier
