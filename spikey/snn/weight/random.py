"""
Random recurrent network generator.
"""
from copy import deepcopy

import numpy as np

from spikey.snn.weight.template import Weight


def generate_masked(fn, mask):
    matrix = np.zeros(mask.shape, dtype=np.float)
    matrix[mask] = fn(np.sum(mask, dtype=np.int))
    return matrix


class Random(Weight):
    """
    Generate recurrent random network.

    Parameters
    ----------
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    NECESSARY_KEYS = deepcopy(Weight.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "force_unidirectional": "bool Whether or not to force matrix unidirectional.",
            "weight_generator": "f(size: int, shape: 2 tuple)->ndarray Function to generate weights.",
            "matrix_mask": "np.bool[inputs+neurons, neurons  OR neurons, neurons] or None. True=generate weights, False=empty.",
            "inh_weight_mask": "ndarray, boolean Matrix showing what synapses are inhibitory",
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._matrix_mask is None:
            input_weights = self._weight_generator((self._n_inputs, self._n_neurons))
            body_weights = self._weight_generator((self._n_neurons, self._n_neurons))
        else:
            mask = self._matrix_mask.astype(np.bool_)
            assert len(mask.shape) == 2, "Mask must be None or 2 dimensional"

            if mask.shape == (self._n_neurons, self._n_neurons):
                input_weights = self._weight_generator(
                    (self._n_inputs, self._n_neurons)
                )
                body_weights = generate_masked(self._weight_generator, mask)
            elif mask.shape == (self._n_inputs + self._n_neurons, self._n_neurons):
                input_weights = generate_masked(
                    self._weight_generator, mask[: self._n_inputs]
                )
                body_weights = generate_masked(
                    self._weight_generator, mask[self._n_inputs :]
                )
            else:
                raise ValueError(
                    "Mask must be None or shaped (n_inputs+n_neurons, n_neurons) or (n_neurons, n_neurons)."
                )

        self._matrix = np.vstack((input_weights, body_weights))

        diagonal = np.arange(self._n_neurons)
        self._matrix[diagonal + self._n_inputs, diagonal] = 0.0

        if kwargs["force_unidirectional"]:
            for x in range(self._n_neurons):
                for y in range(x, self._n_neurons):
                    if (
                        not self._matrix[x + self._n_inputs, y]
                        or not self._matrix[y + self._n_inputs, x]
                    ):
                        continue

                    if np.random.randint(0, 2):
                        self._matrix[x + self._n_inputs, y] = 0.0
                    else:
                        self._matrix[y + self._n_inputs, x] = 0.0

        if self._inh_weight_mask is not None:
            self.inh = np.where(self._inh_weight_mask, -1, 1)
        else:
            self.inh = np.ones(self._matrix.shape)

        self._matrix *= self._max_weight

        self._matrix = np.clip(self._matrix, 0, self._max_weight)
        self._matrix = np.ma.array(self._matrix, mask=(self._matrix == 0), fill_value=0)

    def __mul__(self, multiplier: float) -> np.ndarray:
        return self.matrix * self.inh * multiplier
