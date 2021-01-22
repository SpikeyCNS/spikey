"""
Tests on the neuron dynamics functions.
"""
import unittest

import numpy as np

from spikey.snn.neuron import *


class TestNeuron(unittest.TestCase):
    """
    Neuron dynamics tests.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in [Neuron, RandPotential]:
                with self.subTest(i=obj.__name__):
                    self._get_neuron = self._set_obj(obj)

                    func(self)

        return run_all

    def _set_obj(self, obj):
        """
        Create generator that will render only specific object.
        """

        def _get_neuron(**kwargs):
            np.random.seed(0)

            config = {
                "magnitude": 1,
                "n_neurons": 4,
                "neuron_pct_inhibitory": 0,
                "potential_decay": 0,
                "prob_rand_fire": 0,
                "refractory_period": 5,
                "resting_mv": 0.0,
                "spike_delay": 0,
                "leak_scalar": 0.2,
            }
            config.update(kwargs)

            neuron = obj(**config)
            neuron._generate_spike_shape = lambda *args: np.array(
                [1] + [0] * 9
            ).reshape((-1, 1))

            return neuron

        return _get_neuron

    @run_all_types
    def test_add_potential(self):
        """
        Testing neuron.__iadd__.

        Parameters
        ----------
        incoming_v: np.array(neurons)
            Voltages to increase potentials by.

        Effects
        -------
        Neuron potentials increased by incoming_v
        """
        ## Ensure potentials are updated by incoming voltage.
        N_NEURONS = 100

        for add in range(-10, 10, 2):
            for potential in range(-100, 100, 20):
                neuron = self._get_neuron(n_neurons=N_NEURONS)

                potentials = potential * np.ones(N_NEURONS)
                potentials += np.random.uniform(-1, 1, size=N_NEURONS)

                neuron.potentials = potentials.copy()

                to_add = add * np.ones(N_NEURONS)
                to_add += np.random.uniform(-1, 1, size=N_NEURONS)

                neuron += to_add

                self.assertEqual(list(neuron.potentials), list(potentials + to_add))

        ## Ensure memory safety
        N_NEURONS = 100

        neuron = self._get_neuron(n_neurons=N_NEURONS)

        potentials = np.ones(N_NEURONS)
        potentials += np.random.uniform(-1, 1, size=N_NEURONS)

        neuron.potentials = potentials

        original_to_add = np.ones(N_NEURONS)
        original_to_add += np.random.uniform(-1, 1, size=N_NEURONS)

        to_add = np.copy(original_to_add)

        neuron += to_add

        self.assertListEqual(list(original_to_add), list(to_add))

    @run_all_types
    def test_decay(self):
        """
        Testing neuron.update decay mechanics.

        Settings
        --------
        _potential_decay: float [0, 1]
            Rate to decay neuron potential.

        Effects
        -------
        Neuron potential will decay by a rate relative to _potential_decay.
        """
        ## Assert potential after update is potential * (1 - decay).
        # NOTE: This will fail if templated over
        EXPECTED_RULE = lambda decay_rate, potential: potential * (1 - decay_rate)

        for decay in [1, 0.9, 0.1, 0, -0.5, -1]:
            neuron = self._get_neuron(potential_decay=decay)

            for input_potential in [100, 10, 1, 0, -10]:
                neuron.reset()

                input_potentials = input_potential * np.ones(neuron._n_neurons)
                input_potentials += np.random.uniform(-1, 1, size=neuron._n_neurons)

                neuron.potentials = input_potentials.copy()

                neuron.update()

                self.assertListEqual(
                    list(neuron.potentials),
                    list(EXPECTED_RULE(decay, input_potentials)),
                )

    @run_all_types
    def test_inhibitory(self):
        """
        Test neuron.polarities usage in neuron.reset and neuron.__ge__.

        Settings
        ----------
        _neuron_pct_inhibitory: float [0, 1]
            Percentage of inhibitory neurons.

        Effects
        -------
        The number of inhibitory neurons should be relative to _neuron_pct_inhibitory
        and the polarity of spikes generated is reflected by their polarities.
        """
        ## Assert percentage inhibitory in network is as set in config.
        for percentage in [1, 0.75, 0.25, 0]:
            neuron = self._get_neuron(n_neurons=100000, neuron_pct_inhibitory=percentage)
            neuron.reset()

            real_percentage = (
                np.count_nonzero(neuron.polarities == -1) / neuron.polarities.size
            )

            self.assertAlmostEqual(percentage, real_percentage, places=1)

        ## Assert inhibitory neurons have negative values when spiking.
        neuron = self._get_neuron()
        neuron.reset()

        polarities = np.where(np.arange(neuron._n_neurons) % 2, -1, 1)
        self.assertGreater(np.sum(polarities == -1), 0)

        neuron.polarities = polarities

        spikes = neuron >= neuron._resting_mv

        self.assertListEqual(list(spikes), list(polarities))
        self.assertGreater(np.sum(spikes < 0), 0)

    @run_all_types
    def test_refractory(self):
        """
        Testing neuron.refractory_timers usage in neuron.__ge__ and neuron.update.

        Settings
        --------
        _refractory_period: int
            Duration of refractory period.

        Effects
        -------
        Neuron potential should be -np.inf for the duration of the
        refractory period and _resting_mv immediately afterwards.
        """
        ## Assert potential is -np.inf for duration of refractory period
        ## and _resting_mv immediately afterwards.
        for period in [10, 2, 1, 0, -1]:
            neuron = self._get_neuron(refractory_period=period)
            neuron.reset()

            _ = neuron >= neuron._resting_mv

            for _ in range(period):
                neuron.update()

                self.assertLessEqual(
                    list(neuron.potentials), [-1000] * neuron._n_neurons
                )

            neuron.update()

            self.assertEqual(
                list(neuron.potentials), [neuron._resting_mv] * neuron._n_neurons
            )

    @run_all_types
    def test_reset(self):
        """
        Testing neuron.__reset__.

        Effects
        -------
        All neuron tables should be reset.
        """
        neuron = self._get_neuron()

        EXPECTED_SCHEDULE = [[0 for _ in range(neuron._n_neurons)] for _ in range(10)]

        ## Ensure resets correctly with different start values.
        neuron.refractory_timers = 15 * np.ones(neuron._n_neurons)

        neuron.reset()
        self.assertListEqual(
            list(neuron.potentials), [neuron._resting_mv] * neuron._n_neurons
        )
        self.assertListEqual(list(neuron.refractory_timers), [0] * neuron._n_neurons)
        self.assertListEqual(
            [list(neuron.schedule[i]) for i in range(10)], EXPECTED_SCHEDULE
        )

        ## Ensure resets correctly after an update.
        neuron.refractory_timers = 100 * np.ones(neuron._n_neurons)
        neuron.potentials = 13366 * np.ones(neuron._n_neurons)

        neuron.update()

        neuron.reset()
        self.assertListEqual(
            list(neuron.potentials), [neuron._resting_mv] * neuron._n_neurons
        )
        self.assertListEqual(list(neuron.refractory_timers), [0] * neuron._n_neurons)
        self.assertListEqual(
            [list(neuron.schedule[i]) for i in range(10)], EXPECTED_SCHEDULE
        )

        ## Ensure resets correctly after spiking.
        neuron.potentials = 100 * np.ones(neuron._n_neurons)

        _ = neuron >= 50
        neuron.update()

        neuron.reset()
        self.assertListEqual(
            list(neuron.potentials), [neuron._resting_mv] * neuron._n_neurons
        )
        self.assertListEqual(list(neuron.refractory_timers), [0] * neuron._n_neurons)
        self.assertListEqual(
            [list(neuron.schedule[i]) for i in range(10)], EXPECTED_SCHEDULE
        )

    @run_all_types
    def test_scheduling(self):
        """
        Testing neuron.schedule and neuron._generate_spike_shape usage in
        neuron.__ge__ and neuron.update.

        Settings
        --------
        _generate_spike_shape: func
            Function that generates spike shape.

        Effects
        -------
        Neurons should spike only when scheduled to and should be scheduled
        accirding to the spike shape.
        """
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

    @run_all_types
    def test_spike(self):
        """
        Testing neuron.__ge__.

        Parameters
        ----------
        threshold: float
            Neuron firing thershold.

        Returns
        -------
        spikes: np.array(neurons)
            Spike outputs of neurons.

        Effects
        -------
        Neurons should have spikes schedule if potential above threshold.
        """
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


if __name__ == "__main__":
    unittest.main()
