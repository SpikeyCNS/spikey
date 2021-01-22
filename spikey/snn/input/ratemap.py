"""
Network input dynamics.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.input.template import Input


class RateMap(Input):
    """
    Uniform spike train generator, rates based on enviornment state.

    Parameters
    ----------
    n_inputs: int
        Number of inputs.
    """

    NECESSARY_KEYS = deepcopy(Input.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {"rate_mapping": "list[float] Elementwise State->Rate mapping."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._rate_mapping = np.array(self._rate_mapping)

    def __call__(self):
        """
        Spike output for each input neuron.

        Returns
        -------
        ndarray Spike output for each neuron.
        """
        if not self.values.size:
            return []

        self.network_time += 1
        if self.network_time > self._firing_steps:
            return np.zeros(self.values.shape)

        spikes = np.where(
            np.random.uniform(0, 1, size=self.values.size) <= self.values,
            self._magnitude,
            0.0,
        )

        return spikes * self.polarities

    def update(self, state):
        """
        Update input settings.

        Parameters
        ----------
        state: list of float
            Discretized enviornment state.
        """
        self.network_time = 0 if self._firing_steps != -1 else -1000000

        if isinstance(state, (int, float)):
            state = np.array([state])
        else:
            state = np.array(state)

        state = self._rate_mapping[np.int_(state)]

        if not len(state) or self._n_inputs % len(state):
            raise ValueError(f"N_INPUTS must be a multiple of state: {len(state)}!")

        self.values = np.ravel(
            np.array([state for _ in range(self._n_inputs // len(state))])
        )
