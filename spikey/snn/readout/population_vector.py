"""
Population vector coding readout.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.readout.template import Readout


class PopulationVector(Readout):
    """
    Population vector coding readout.
    Ordinal control.
    """

    NECESSARY_KEYS = deepcopy(Readout.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "n_actions": "int Number of action groups.",
        }
    )

    def __call__(self, output_spike_train: np.ndarray) -> np.ndarray:
        if self._n_outputs == 0:
            return np.zeros(self._n_actions)

        spikes = np.where(output_spike_train, 1, 0)

        T_gamma, V_gamma = 50, 20
        gamma = lambda t: (np.e ** (-t / T_gamma) - np.e ** (-t / V_gamma)) / (
            T_gamma - V_gamma
        )
        decay_kernel = np.vectorize(gamma)(np.arange(spikes.shape[0])).reshape((-1, 1))

        firing_rates = np.sum(spikes * decay_kernel, axis=0)

        group_size = self._n_outputs // self._n_actions

        p = [
            np.sum(firing_rates[i * group_size : (i + 1) * group_size])
            for i in range(self._n_actions)
        ]
        p = np.array(p)

        if np.sum(p) != 0:
            actions = p / np.sum(p)
        else:
            actions = np.ones(p.shape) / p.size

        return actions
