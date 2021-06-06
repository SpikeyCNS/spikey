"""
Tests for snn.Readout.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import readout


class TestReadout(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Readout.
    """

    TYPES = [
        readout.Threshold,
        readout.NeuronRates,
        readout.PopulationVector,
        readout.TopAction,
    ]
    BASE_CONFIG = {
        "n_outputs": 2,
        "magnitude": 1,
        "action_threshold": 0.5,
        "output_range": [0, 1],
        "n_actions": 1,
    }

    @ModuleTest.run_all_types
    def test_usage(self):
        readout = self.get_obj()
        readout.reset()

        log_shape = (10, self.BASE_CONFIG["n_outputs"])
        action = readout(np.ones(log_shape))
        action = readout(np.random.uniform(0, 1, size=log_shape))
        action = readout(np.random.uniform(-5, 5, size=log_shape))


if __name__ == "__main__":
    unittest.main()
