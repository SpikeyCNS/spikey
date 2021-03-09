"""
Tests for snn.Input.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import input


def get_values(state):
    return np.zeros(shape=len(state))


class TestInput(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Input.
    """

    TYPES = [input.RateMap]  # , input.StaticMap, input.RBF]
    BASE_CONFIG = {
        "n_inputs": 2,
        "get_values": get_values,
        "magnitude": 1,
        "firing_steps": -1,
        "rate_mapping": [0, 1],
        "input_pct_inhibitory": 0,
        "state_rate_map": np.arange(10) / 10,
    }

    @ModuleTest.run_all_types
    def test_len(self):
        for n_inputs in [1, 100]:
            inputs = self.get_obj(n_inputs=n_inputs)
            self.assertEqual(len(inputs), n_inputs)


if __name__ == "__main__":
    unittest.main()
