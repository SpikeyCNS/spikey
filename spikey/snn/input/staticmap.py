"""
Custom state - input firings mapping.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.input.template import Input


class StaticMap(Input):
    """
    Custom state - input firings mapping.

    Parameters
    ----------
    n_inputs: int
        Number of inputs.
    """

    NECESSARY_KEYS = deepcopy(Input.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "mapping": "dict[tuple]->ndarray[processing_time, n_inputs] State to fires mapping.."
        }
    )

    def __call__(self):
        """
        Spike output for each input neuron.

        Returns
        -------
        ndarray Spike output for each neuron.
        """
        output = np.array(self._mapping[self.values])

        if len(output.shape) > 1:
            spikes = [value * self._magnitude for value in output[self.time]]
        else:
            spikes = [value * self._magnitude for value in output]

        self.time += 1

        return np.array(spikes) * self.polarities

    def update(self, state):
        """
        Update input settings.

        Parameters
        ----------
        state: list of float
            Discretized enviornment state.
        """
        self.time = 0

        self.values = tuple(state)
