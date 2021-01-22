"""
Tests on the population readout.
"""
import unittest

import numpy as np

from spikey.snn.readout import *


## TODO test for neuronrates


class TestReadout(unittest.TestCase):
    """
    Unit test for population readout.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in [
                Threshold,
                NeuronRates,
                PopulationVector,
            ]:
                with self.subTest(i=obj.__name__):
                    self._get_readout = self._set_obj(obj)

                    func(self)

        return run_all

    def _set_obj(self, obj):
        """
        Create readout that will render only specific object.
        """

        def _get_readout(**kwargs):
            np.random.seed(0)

            config = {
                "n_outputs": 2,
                "magnitude": 1,
                "action_threshold": 0.5,
                "output_range": [0, 1],
                "n_actions": 1,
                "n_pools": 0,
            }
            config.update(kwargs)

            readout = obj(**config)

            return readout

        return _get_readout

    @run_all_types
    def test_call(self):
        """
        Testing readout.__call__.

        Settings
        --------
        _magnitude: float
            Spike magnitude.

        Returns
        -------
        float Action.
        """
        ## Ensure does not crash
        for N_OUTPUTS in range(2, 5):
            readout = self._get_readout(n_outputs=N_OUTPUTS)

            for rate in np.arange(0, 1, 0.1):
                LENGTH = 100
                spike_log = np.zeros(shape=(LENGTH, N_OUTPUTS))
                spike_log[: int(rate * LENGTH)] = 1

                action = readout(spike_log)

                self.assertTrue(action is not None)

        ## Ensure returns empty list with 0 outputs
        readout = self._get_readout(n_outputs=0)

        for rate in np.arange(0, 1, 0.1):
            action = readout([[] for _ in range(100)])

            self.assertIn(action, [0, (), [], np.array([])])

        """
        ## Ensure output_range is respected
        for output_range in [[0, 1], [-1, 1], [-.616, 0]]:
            actions = []

            for rate in np.arange(0, 1.1, .1):
                LENGTH = 100
                spike_log = np.zeros(shape=(LENGTH, 2))
                spike_log[:int(rate * LENGTH), 0] = 1
                spike_log[int(rate * LENGTH * 1.5):, 1] = 1

                readout = self._get_readout(output_range=output_range, n_outputs=2)

                readout(np.array([[0, 0], [1, 1]])) ## for mean threshold
                action = readout(spike_log)

                if not isinstance(action, (int, float)):
                    action = action[0]

                actions.append(action)

                self.assertIn(action, output_range)
        """

        ## Ensure memory safe
        LENGTH = 100
        original_spike_log = np.zeros(shape=(LENGTH, 2))
        original_spike_log[: int(rate * LENGTH), 0] = 1
        original_spike_log[int(rate * LENGTH * 1.5) :, 1] = 1

        spike_log = np.copy(original_spike_log)

        readout = self._get_readout(n_outputs=2)

        readout(np.array([[0, 0], [1, 1]]))
        action = readout(spike_log)

        self.assertListEqual(
            list(np.ravel(original_spike_log)), list(np.ravel(spike_log))
        )


if __name__ == "__main__":
    unittest.main()
