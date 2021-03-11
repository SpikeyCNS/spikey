"""
Tests for snn.Synapse.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import synapse


class FakeWeight:
    def __init__(self, n_inputs, n_neurons, max_weight=1):
        self._matrix = np.ma.array(np.ones((n_inputs+n_neurons, n_neurons)))
        self._max_weight = max_weight

    @property
    def matrix(self):
        return self._matrix.data

    def __mul__(self, multiplier: np.ndarray) -> np.float:
        return self.matrix * multiplier



class TestSynapse(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Synapse.
    """

    TYPES = [synapse.RLSTDPET, synapse.LTP]
    BASE_CONFIG = {
        "w": FakeWeight(10, 50),
        "n_inputs": 10,
        "n_neurons": 50,
        "stdp_window": 200,
        "learning_rate": .05,
        "max_weight": 1,
        "trace_decay": .1,
    }

    @ModuleTest.run_all_types
    def test_usage(self):
        n_neurons = self.BASE_CONFIG['n_neurons']
        synapses = self.get_obj(n_inputs=0, w=FakeWeight(0, n_neurons))
        synapses.reset()

        polarities = np.zeros(n_neurons)
        pre_fires = np.random.uniform(size=n_neurons) <= .08
        post_fires = np.matmul(synapses.weights.matrix, pre_fires) >= 2

        synapses.update(np.vstack((post_fires, pre_fires)), polarities)
        synapses.update(pre_fires.reshape((-1, 1)), polarities)
        synapses.update(np.array([]), polarities)



if __name__ == "__main__":
    unittest.main()
