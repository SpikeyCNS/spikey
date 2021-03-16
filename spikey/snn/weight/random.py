"""
Randomly generated network.
The data structure to generate and manage connections between neurons.
Contains generation, arithmetic and get operations.
Updates are handled in spikey.snn.Synapse objects.
"""
import numpy as np
from spikey.module import Key
from spikey.snn.weight.template import Weight


def generate_masked(fn, mask):
    matrix = np.zeros(mask.shape, dtype=np.float)
    matrix[mask] = fn(np.sum(mask, dtype=np.int))
    return matrix


class Random(Weight):
    """
    Randomly generated network.
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

    Examples
    --------

    .. code-block:: python

        config = {
            "n_inputs": 1,
            "n_neurons": 10,
            "max_weight": 3,
            "force_unidirectional": True,
            "weight_generator": lambda *a, **kw: np.random.uniform(0, 3, *a, **kw),
            "matrix_mask": np.random.uniform(size=(1+10, 10)) <= .2,
        }
        w = Random(**config)

        in_volts = w * np.ones(config['n_neurons'])

    .. code-block:: python

        class network_template(Network):
            keys = {
                "n_inputs": 1,
                "n_neurons": 10,
                "max_weight": 3,
                "force_unidirectional": True,
                "weight_generator": lambda *a, **kw: np.random.uniform(0, 3, *a, **kw),
                "matrix_mask": np.random.uniform(size=(1+10, 10)) <= .2,
            }
            parts = {
                "weights": Random
            }
    """

    NECESSARY_KEYS = Weight.extend_keys(
        [
            Key(
                "force_unidirectional",
                "bool Whether or not to force matrix unidirectional.",
                bool,
                default=False,
            ),
            Key(
                "weight_generator",
                "f(size: int, shape: 2 tuple)->ndarray Function to generate weights.",
            ),
            Key(
                "matrix_mask",
                "np.bool[inputs+neurons, neurons  OR neurons, neurons] or None. True=generate weights, False=empty.",
                (np.ndarray, type(None)),
            ),
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._matrix_mask is None:
            input_weights = self._weight_generator((self._n_inputs, self._n_neurons))
            body_weights = self._weight_generator((self._n_neurons, self._n_neurons))
        else:
            mask = self._matrix_mask.astype(np.bool_)

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
                self._assert_matrix_shape(self._matrix_mask, key="matrix_mask")

        self._matrix = np.vstack((input_weights, body_weights))

        diagonal = np.arange(self._n_neurons)
        self._matrix[diagonal + self._n_inputs, diagonal] = 0.0

        if self._force_unidirectional:
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

        self._matrix *= self._max_weight

        self._matrix = np.clip(self._matrix, 0, self._max_weight)
        self._matrix = np.ma.array(self._matrix, mask=(self._matrix == 0), fill_value=0)

        self._assert_matrix_shape(self._matrix, key="matrix")
