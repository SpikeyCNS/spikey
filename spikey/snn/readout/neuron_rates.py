"""
Individual neuron rates population readout.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.readout.template import Readout


class NeuronRates(Readout):
    """
    Individual neuron rates population readout.
    """

    NECESSARY_KEYS = deepcopy(Readout.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "n_pools": "int Number of groups to put neurons into. 0 pools means each neuron separate output.",
        }
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._n_pools == 0:
            self._n_pools = self._n_outputs

    def __call__(self, output_spike_train: np.bool) -> float:
        if self._n_outputs == 0:
            return 0

        idx = np.linspace(0, self._n_outputs, self._n_pools + 1).astype(np.int)
        pools = [output_spike_train[idx[i] : idx[i + 1]] for i in range(self._n_pools)]
        return np.mean(pools, axis=(1, 2))
