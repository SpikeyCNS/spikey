"""
Tests for snn.Neuron.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import neuron


class TestNeuron(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Neuron.
    """

    TYPES = [neuron.Neuron, neuron.RandPotential]
    BASE_CONFIG = {
        "magnitude": 1,
        "n_neurons": 4,
        "firing_threshold": 2,
        "neuron_pct_inhibitory": 0,
        "potential_decay": 0,
        "prob_rand_fire": 0,
        "refractory_period": 5,
        "resting_mv": 0.0,
        "spike_delay": 0,
        "potential_noise_scale": .1,
    }

"""
    @run_all_types
    def test_scheduling(self):
        ## Ensure given spike shape sizes are valid.
        get_spike_shape = lambda *a: np.array([1] + [0] * length).reshape((-1, 1))

        neuron = self._get_neuron()
        neuron._generate_spike_shape = get_spike_shape

        for length in range(10):
            neuron.reset()

            neuron.potentials = 100 * np.ones(neuron._n_neurons)

            for value in get_spike_shape():
                spikes = neuron >= 1
                neuron.update()

                self.assertListEqual(list(spikes), [value] * neuron._n_neurons)

        ## Ensure given spike shapes values are valid.
        get_spike_shape = lambda *a: np.array(
            [np.sin(i + j) for j in range(10)]
        ).reshape((-1, 1))

        neuron = self._get_neuron()
        neuron._generate_spike_shape = get_spike_shape

        for i in range(10):
            neuron.reset()

            neuron.potentials = np.ones(neuron._n_neurons)

            for value in get_spike_shape():
                spikes = neuron >= 1
                neuron.update()

                self.assertListEqual(list(spikes), list(value) * neuron._n_neurons)

        ## Ensure spikes do not get overwritten.
        N_NEURONS = 100
        SCHEDULE_LEN = 20

        EXPECTED = [i * 0.5 + 0.5 for i in range(SCHEDULE_LEN)]
        get_spike_shape = lambda *a: np.array(
            [0.5 for j in range(SCHEDULE_LEN)]
        ).reshape((-1, 1))

        neuron = self._get_neuron(n_neurons=N_NEURONS, refractory_period=0)
        neuron._generate_spike_shape = get_spike_shape

        neuron.reset()

        for i, value in enumerate(EXPECTED):
            neuron.potentials = np.ones(neuron._n_neurons)
            spikes = neuron >= 1
            neuron.update()

            self.assertListEqual(list(spikes), [value] * N_NEURONS)

        threshold = 50

        N_NEURONS = 100

        ## Ensure that the neuron updates correctly after firing.

        # - refractory_period > 0
        neuron = self._get_neuron(n_neurons=N_NEURONS, refractory_period=1)

        for potential in [threshold, threshold + 100]:
            neuron.reset()

            neuron.potentials = potential * np.ones(N_NEURONS)

            _ = neuron >= threshold
            neuron.update()

            self.assertLessEqual(list(neuron.potentials), [-1000] * N_NEURONS)

        # - refractory_period = 0
        neuron = self._get_neuron(n_neurons=N_NEURONS, refractory_period=0)

        for potential in [threshold, threshold + 100]:
            neuron.reset()

            neuron.potentials = potential * np.ones(N_NEURONS)

            _ = neuron >= threshold
            neuron.update()

            self.assertEqual(list(neuron.potentials), [neuron._resting_mv] * N_NEURONS)

        ## Ensure spikes based on schedule.
        def get_spike_shape(*_):
            spike_shape = np.zeros(shape=(10, 1))
            spike_shape[delay] = 1
            return spike_shape

        neuron = self._get_neuron(n_neurons=N_NEURONS)
        neuron._generate_spike_shape = get_spike_shape

        for delay in range(10):
            neuron.reset()

            neuron.potentials = 100 * np.ones(N_NEURONS)

            for _ in range(delay):
                spikes = neuron >= 100
                self.assertEqual(list(spikes), [0] * N_NEURONS)

                neuron.update()

            spikes = neuron >= 100
            self.assertEqual(list(spikes), [1] * N_NEURONS)
"""

if __name__ == "__main__":
    unittest.main()
