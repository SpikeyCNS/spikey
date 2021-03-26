"""
Tests for snn.Weight.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import weight


def weight_generator(shape):
    matrix = np.random.uniform(0, 9999, size=shape)
    return np.ma.array(matrix, mask=(matrix == 0), fill_value=0)


class TestWeight(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Weight.
    """

    TYPES = [weight.Manual, weight.Random]
    BASE_CONFIG = {
        "n_inputs": 10,
        "n_neurons": 10,
        "max_weight": 1,
        "force_unidirectional": False,
        "matrix": weight_generator((10 + 10, 10)),
        "weight_generator": weight_generator,
        "matrix_mask": None,
    }

    def _check_matrix_types(self, weights):
        self.assertIsInstance(weights._matrix, np.ndarray)
        self.assertIsInstance(weights.matrix, np.ndarray)
        self.assertTrue(hasattr(weights._matrix, "mask"))
        self.assertTrue(not hasattr(weights.matrix, "mask"))
        self.assertEqual(weights._matrix.fill_value, 0)

    def _assert_clipped(self, weights):
        self.assertTrue(
            np.all(weights.matrix >= 0)
            & np.all(weights.matrix <= self.BASE_CONFIG["max_weight"])
        )

    @ModuleTest.run_all_types
    def test_arith(self):
        w_shape = (
            self.BASE_CONFIG["n_inputs"] + self.BASE_CONFIG["n_neurons"],
            self.BASE_CONFIG["n_neurons"],
        )

        weights = self.get_obj(
            max_weight=2, matrix=np.ma.array(np.ones(w_shape)), weight_generator=np.ones
        )
        self.assertTrue(np.mean(weights._matrix.mask) <= 0.05)
        weights += np.ones((w_shape[0], 1))
        self.assertTrue(np.mean(weights._matrix == 2) >= 0.95)
        self.assertTrue(np.mean(weights.matrix == 2) >= 0.95)

        weights = self.get_obj(
            max_weight=2, matrix=np.zeros(w_shape), weight_generator=np.zeros
        )
        self.assertTrue(np.all(weights._matrix.mask))
        weights += np.ones((w_shape[0], 1))
        self.assertTrue(np.all(weights.matrix == 0))

    @ModuleTest.run_all_types
    def test_usage(self):
        weights = self.get_obj()
        self._check_matrix_types(weights)
        self._assert_clipped(weights)

        spike_shape = (self.BASE_CONFIG["n_inputs"] + self.BASE_CONFIG["n_neurons"], 1)
        weights * np.random.uniform(0, 1, spike_shape)
        weights / np.random.uniform(0, 1, spike_shape)
        weights + np.random.uniform(0, 1, spike_shape)
        weights - np.random.uniform(0, 1, spike_shape)

        weights *= np.random.uniform(0, 1, spike_shape)
        weights /= np.random.uniform(0, 1, spike_shape)
        weights += np.random.uniform(0, 1, spike_shape)
        weights -= np.random.uniform(0, 1, spike_shape)

        weights.matrix
        self._check_matrix_types(weights)
        self._check_matrix_types(weights.copy())

        layers = [np.ones((10, 10)), np.ones((4, 4)), np.ones((1, 1))]
        weights = self.get_obj(**{"matrix": layers, "matrix_mask": layers})
        self._check_matrix_types(weights)
        self._assert_clipped(weights)


if __name__ == "__main__":
    unittest.main()
