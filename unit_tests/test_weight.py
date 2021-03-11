"""
Tests for snn.Weight.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import weight


class TestWeight(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Weight.
    """

    TYPES = [weight.Manual, weight.Random]
    BASE_CONFIG = {
        'n_inputs': 10,
        'n_neurons': 10,
        'max_weight': 1,
        'force_unidirectional': False,
        'matrix': np.random.uniform(0, 1, size=(10+10, 10)),
        'weight_generator': lambda shape: np.random.uniform(0, 1, size=shape),
        'matrix_mask': None,
    }

    @ModuleTest.run_all_types
    def test_usage(self):
        weights = self.get_obj()


if __name__ == "__main__":
    unittest.main()
