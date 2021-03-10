"""
Tests for snn.Neuron.
"""
import unittest
from unittest import mock
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import neuron


class TestNeuron(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Neuron.
    """

    TYPES = [neuron.Neuron, neuron.RandPotential]
    BASE_CONFIG = {
        "magnitude": 5.6,
        "n_neurons": 4,
        "firing_threshold": 2,
        "neuron_pct_inhibitory": 0,
        "potential_decay": .25,
        "prob_rand_fire": 0,
        "refractory_period": 5,
        "resting_mv": .1,
        "spike_delay": 0,
        "potential_noise_scale": 0,
    }

    @ModuleTest.run_all_types
    def test_polarity(self):
        neurons = self.get_obj()
        neurons.reset()
        self.assertTrue(hasattr(neurons, "polarities"))
        self.assertIsInstance(neurons.polarities, np.ndarray)

        neurons.polarities = type(
            "polarity", (object,), {"__rmul__": mock.Mock(return_value=0)}
        )()
        spikes = neurons()
        neurons.polarities.__rmul__.assert_called()

    @ModuleTest.run_all_types
    def test_loop(self):
        firing_threshold = 4
        neurons = self.get_obj(firing_threshold=firing_threshold, potential_decay=0)
        neurons.reset()
        # assert potentials is resting
        # assert refactory timers 0

        for time in range(firing_threshold + 4):
            spikes = neurons()
            self.assertIsInstance(spikes, np.ndarray)
            neurons += np.ones(self.BASE_CONFIG['n_neurons'])

            if time < firing_threshold:
                self.assertTrue(np.all(spikes == 0))
                if hasattr(neurons, 'refactory_timers'):
                    self.assertTrue(np.all(neurons.refactory_timers == 0))
            elif time == firing_threshold:
                self.assertTrue(np.all(spikes == self.BASE_CONFIG['magnitude']))
                self.assertTrue(np.all(neurons.potentials < firing_threshold))
                if hasattr(neurons, 'refactory_timers'):
                    self.assertTrue(np.all(neurons.refactory_timers != 0))
            else:
                self.assertTrue(np.all(spikes == 0))
                if hasattr(neurons, 'refactory_timers'):
                    self.assertTrue(np.all(neurons.refactory_timers != 0))

if __name__ == "__main__":
    unittest.main()
