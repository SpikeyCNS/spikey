"""
Tests for snn.Input.
"""
import unittest
from unittest import mock
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import input


N_STATES = 10
PROCESSING_TIME = 100

state_rate_map = np.arange(N_STATES) / N_STATES
state_spike_map = np.random.uniform(
    (N_STATES, PROCESSING_TIME)
) <= state_rate_map.reshape((-1, 1))


def get_values(state):
    return np.zeros(shape=len(state))


class TestInput(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Input.
    """

    TYPES = [input.RateMap, input.StaticMap]
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
        inputs.reset()
        self.assertTrue(hasattr(inputs, "polarities"))
        self.assertIsInstance(inputs.polarities, np.ndarray)

        inputs.polarities = type(
            "polarity", (object,), {"__rmul__": mock.Mock(return_value=0)}
        )()
        inputs.update(5)
        spikes = inputs()
        inputs.polarities.__rmul__.assert_called()

    @ModuleTest.run_all_types
    def test_update(self):
        inputs = self.get_obj()
        inputs.reset()

        inputs.update(2)
        inputs.update((0,))
        inputs.update(np.array([1]))

    @ModuleTest.run_all_types
    def test_loop(self):
        inputs = self.get_obj()
        inputs.reset()

        for state in range(3):
            inputs.update(state)
            self.assertEqual(inputs.network_time, 0)

            for time in range(PROCESSING_TIME):
                spikes = inputs()
                self.assertEqual(inputs.network_time, time + 1)
                self.assertIsInstance(spikes, np.ndarray)


if __name__ == "__main__":
    unittest.main()
