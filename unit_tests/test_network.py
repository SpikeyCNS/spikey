"""
Tests for snn.Network.
"""
import unittest
from unit_tests import ModuleTest
from copy import deepcopy
import numpy as np
from spikey.snn import network


class FakeBase:
    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        return lambda *a, **kw: None


class FakeInput(FakeBase):
    def __call__(self):
        return np.ones(10)


class FakeNeuron(FakeBase):
    def __call__(self):
        return np.ones(20)

    def __iadd__(self, other):
        return self


class FakeWeight(FakeBase):
    def __init__(self, **kwargs):
        self.weights = np.ones(
            (kwargs["n_inputs"] + kwargs["n_neurons"], kwargs["n_neurons"])
        )

    def __mul__(self, other):
        return self.weights * other


class FakeSynapse(FakeBase):
    def __init__(self, w, **kwargs):
        self._stdp_window = 10


class FakeReadout(FakeBase):
    def __call__(self, spikes):
        return None


class FakeRewarder(FakeBase):
    def __call__(self, *a, **kw):
        return 1


def continuous_rwd_action(*a, **kw):
    return 0


class TestNetwork(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Network.
    """

    TYPES = [network.Network, network.RLNetwork, network.ContinuousRLNetwork]
    BASE_CONFIG = {
        "inputs": FakeInput,
        "neurons": FakeNeuron,
        "weights": FakeWeight,
        "synapses": FakeSynapse,
        "readout": FakeReadout,
        "rewarder": FakeRewarder,
        "n_inputs": 10,
        "n_neurons": 20,
        "n_outputs": 10,
        "processing_time": 10,
        "continuous_rwd_action": continuous_rwd_action,
    }

    @ModuleTest.run_all_types
    def test_init(self):
        network_type = type(self.get_obj())

        class network_template(network_type):
            parts = deepcopy(self.BASE_CONFIG)
            keys = deepcopy(self.BASE_CONFIG)
            del keys["inputs"], keys["neurons"]
            keys.update(
                {
                    "n_inputs": 10,
                    "n_neurons": 20,
                    "n_outputs": 30,
                }
            )

        n_inputs = 11
        network = network_template(n_inputs=n_inputs)
        self.assertEqual(network._n_inputs, n_inputs)
        self.assertEqual(network._n_neurons, network_template.keys["n_neurons"])
        self.assertEqual(network._n_outputs, network_template.keys["n_outputs"])

    @ModuleTest.run_all_types
    def test_usage(self):
        network = self.get_obj()
        network.reset()

        for state in [0, 1, 10, (1, 2), np.array([1, 2, 3, 4])]:
            action = network.tick(state)
            if hasattr(network, "reward"):
                reward = 1000000
                reward_real = network.reward(state, action, None, reward=reward)
                self.assertEqual(reward, reward_real)


if __name__ == "__main__":
    unittest.main()
