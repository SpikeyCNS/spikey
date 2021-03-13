"""
Tests for snn.Network.
"""
import unittest
from unit_tests import ModuleTest
from spikey.snn import network


class FakeBase:
    def __init__(self, *a, **kw):
        pass

    def __getattribute__(self, *a):
        return lambda *a, **kw: None


class FakeInput(FakeBase):
    pass


class FakeNeuron(FakeBase):
    pass


class FakeWeight(FakeBase):
    pass


class FakeSynapse(FakeBase):
    pass


class FakeReadout(FakeBase):
    pass


class FakeRewarder(FakeBase):
    pass


def continuous_rwd_action(*a, **kw):
    return 0


class TestInput(unittest.TestCase, ModuleTest):
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
        # Test base init

        # Test class as template, w/ 1 param overriden by kwargs
        # Ensure params right on each
        pass

    """
    @ModuleTest.run_all_types
    def test_usage(self):
        network = self.get_obj()
        network.reset()

        for state in [0, 1, 10, (1, 2), np.array([1, 2, 3, 4])]:
            action = network.tick(state)
            if hasattr(network, 'reward'):
                reward = network.reward(state, action)
    """


if __name__ == "__main__":
    unittest.main()
