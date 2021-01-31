"""
Testing the spiking neural network.
"""
import unittest
from unittest import mock

import numpy as np

from spikey.snn import RLNetwork, FlorianSNN
from spikey.snn import *


class ParameterizedSNN(RLNetwork):
    """
    Generic Spiking Neural Network
    """

    network_config = {
        "firing_threshold": 10,
        "processing_time": 1,
    }

    input_config = {
        "magnitude": 1,
        "firing_steps": -1,
        "input_pct_inhibitory": 0,
    }

    neuron_config = {
        "potential_decay": 0,
        "neuron_pct_inhibitory": 0,
        "refractory_period": 0,
        "resting_mv": 0,
        "n_neurons": 100,
        "prob_rand_fire": 0.0,
        "spike_delay": 0,
    }

    synapse_config = {
        "learning_rate": 0.01,
        "stdp_window": 4,
        "max_update": 0.5,
        "max_weight": 1,
        "trace_decay": 1,
    }

    config = {**network_config, **input_config, **neuron_config, **synapse_config}

    class FakeInput:
        def __init__(*args, **kwargs):
            pass

        def __call__(*args, **kwargs):
            return np.ones(input_config["n_inputs"])

        def update(*args, **kwargs):
            pass

    class FakeReadout:
        def __init__(*args, **kwargs):
            pass

        def __call__(*args, **kwargs):
            return 0

    FakeNeuron = mock.MagicMock()
    FakeNeuron.__ge__ = mock.MagicMock(return_value=np.zeros(shape=100))
    FakeSynapse = mock.MagicMock()
    inputs = mock.Mock(return_value=FakeInput)
    neurons = mock.Mock(return_value=FakeNeuron)  # Neuron
    synapses = mock.Mock(return_value=FakeSynapse)
    readout = mock.Mock(return_value=FakeReadout)

    _template_parts = {
        "inputs": inputs,
        "neurons": neurons,
        "synapses": synapses,
        "weights": lambda *args, **kwargs: np.zeros(
            shape=(
                ParameterizedSNN.config["n_neurons"]
                + ParameterizedSNN.config["n_inputs"],
                ParameterizedSNN.config["n_neurons"],
            )
        ),
        "readout": readout,
        "modifiers": None,
    }

    def __init__(self, set_input_rates=None, n_inputs=-1, n_outputs=-1, **kwargs):
        super().__init__(set_input_rates, n_inputs, n_outputs, **kwargs)
        self.spike_log = []


## For use in testing network.tick.
output = []


class TestNetwork(unittest.TestCase):
    """
    Assorted tests for the snn.
    """

    @staticmethod
    def _get_network(*args, **kwargs):
        np.random.seed(0)

        config = {
            "n_inputs": 0,
            "n_outputs": 0,
            "set_input_rates": lambda state: np.array([]),
            "choose_action": lambda outputs: 0,
        }
        config.update(kwargs)
        network = ParameterizedSNN(**config)

        return network

    def test_init(self):
        ## Ensure parts are read // overwritten correctly.
        PARTS = {
            "synapse": [mock.Mock(), mock.Mock()],
            "neuron": [mock.Mock(), mock.Mock()],
            "weight": [mock.Mock(), mock.Mock(), mock.Mock()],
            "input": [mock.Mock(), mock.Mock()],
        }

        for key, values in PARTS.items():
            for value in values:
                network = self._get_network(**{key: value})

                self.assertEqual(value, network.parts[key])
                value.assert_called_once()

    def test_reset(self):
        """
        Testing network.reset.

        Effects
        -------
        All pieces of network should be reset.
        """
        ## Ensure time and neurons reset.
        network = self._get_network()

        network.reset()

        network.neurons.reset.assert_called_once()
        network.synapses.reset.assert_called_once()

        ## Ensure time resets.
        network = self._get_network(processing_time=0)
        network.reset()

        for _ in range(10):
            network.tick([])

        network.reset()
        self.assertEqual(network.internal_time, 0)

    def test_tick(self):
        """
        Testing network.tick.

        Parameters
        ----------
        state: list(float)
            Discretized enviornment state.

        Settings
        --------
        _processing_time: int
            How many updates per tick the network will process.
        choose_action: func
            The function that chooses action based on outputs neurons.

        Returns
        -------
        An action selected by the number of times each output neruon fires.

        Effects
        -------
        Neuron and synapse should be updated every network update, neuron
        potentials should be increased relative to neuron spikes and
        synapse weights.
        """
        ## Ensure time increases and neuron.update and synapse.update
        ## called processing_time times.
        for processing_time in [0, 1]:
            network = self._get_network(n_neurons=0, processing_time=processing_time)

            network.reset()

            network.neurons.__ge__ = lambda *a: np.ones(shape=0)
            network.synapses.weights = np.array([[]])

            network.neurons.__iadd__ = lambda self, value: self
            network.tick([])

            self.assertEqual(network.internal_time, processing_time)

            if processing_time == 1:
                network.neurons.update.assert_called_once()
                network.synapses.update.assert_called_once()
            elif processing_time:
                network.neurons.update.assert_called()
                network.synapses.update.assert_called()
            else:
                network.neurons.update.assert_not_called()
                network.synapses.update.assert_not_called()

        ## Ensure output selection used.
        network = self._get_network(processing_time=0)
        network.reset()

        network._n_outputs = 60
        network.readout = lambda o: len(o) == 60

        self.assertTrue(network.tick([]))

        ## Ensure weight updates calculated and applied correctly.
        N_NEURONS = 100

        WEIGHTS = [  # (weight_matrix, expected)
            (0.5 * np.ones(shape=(N_NEURONS, N_NEURONS)), 0.5 * np.ones(N_NEURONS)),
            (np.zeros(shape=(N_NEURONS, N_NEURONS)), np.zeros(N_NEURONS)),
            (
                np.where(
                    np.arange(0, N_NEURONS ** 2).reshape((N_NEURONS, N_NEURONS)) % 2,
                    1,
                    0,
                ),
                np.array([i % 2 for i in range(N_NEURONS)]),
            ),
        ]

        def __iadd__(self, new_potentials):
            global output  # Defined outside of TestNetwork.
            output = new_potentials
            return self

        for weights, expected in WEIGHTS:
            for mult in [-2, 0, 2, 100]:
                network = self._get_network(
                    processing_time=1,
                    n_inputs=0,
                    n_neurons=N_NEURONS,
                    firing_threshold=mult,
                )
                network.reset()

                network.synapses.configure_mock(w=weights)

                network.neurons.configure_mock(polarities=None)
                network.neurons.__ge__ = lambda *a: mult * np.ones(shape=N_NEURONS)
                network.neurons.__iadd__ = __iadd__

                network.tick([])

                self.assertListEqual(list(output), list(mult * N_NEURONS * expected))

        ## Ensure memory safety
        network = self._get_network(processing_time=1)
        network.reset()

        original_state = [0, 1]
        state = [0, 1]

        network.tick(state)

        self.assertListEqual(original_state, state)


if __name__ == "__main__":
    unittest.main()
