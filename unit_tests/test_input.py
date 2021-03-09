"""
Tests for snn.Input.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import input


N_STATES = 10
PROCESSING_TIME = 100

state_rate_map = np.arange(N_STATES) / N_STATES
state_spike_map = np.random.uniform((N_STATES, PROCESSING_TIME)) <= state_rate_map.reshape((-1, 1))

def get_values(state):
    return np.zeros(shape=len(state))


class TestInput(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Input.
    """

    TYPES = [input.RateMap, input.StaticMap, input.RBF]
    BASE_CONFIG = {
        "n_inputs": 2,
        "magnitude": 1,
        "input_firing_steps": -1,
        "input_pct_inhibitory": 0,
        "state_rate_map": state_rate_map,
        "state_spike_map": state_spike_map,
    }

    @ModuleTest.run_all_types
    def test_len(self):
        for n_inputs in [1, 100]:
            inputs = self.get_obj(n_inputs=n_inputs)
            self.assertEqual(len(inputs), n_inputs)

    @ModuleTest.run_all_types
    def test_polarity(self):
        inputs = self.get_obj()
        self.assertTrue(hasattr(inputs, 'polarities'))
        self.assertIsInstance(inputs.polarities, np.ndarray)


if __name__ == "__main__":
    unittest.main()
