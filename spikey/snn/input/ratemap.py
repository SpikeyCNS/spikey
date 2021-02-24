"""
Uniform spike train generator with rates based on environment state.
"""
from copy import deepcopy
import numpy as np

from spikey.module import Key
from spikey.snn.input.template import Input


class RateMap(Input):
    """
    Uniform spike train generator with rates based on environment state.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    processing_time = 10
    config = {
        "n_inputs": 10,
        "magnitude": 2,
        "input_firing_steps": -1,
        "input_pct_inhibitory": 0.2,
        "state_rate_map": [.0, .8],
    }
    input = RateMap(**config)
    input.reset()
    env = Logic(preset='XOR')

    state = env.reset()
    for step in range(10):
        input.update(state)

        for _ in range(processing_time)
            in_fires = input.__call__()

        state, _, done, __ = env.update(0)

        if done:
            break
    ```

    ```python
    class network_template(Network):
        config = {
            "n_inputs": 10,
            "magnitude": 2,
            "input_firing_steps": -1,
            "input_pct_inhibitory": 0.2,
            "state_rate_map": [.0, .8],
        }
        _template_parts = {
            "inputs": RateMap
        }
    ```
    """

    NECESSARY_KEYS = Input.extend_keys(
        [Key("state_rate_map", "list[float] Elementwise State->Rate map.")]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._state_rate_map = np.array(self._state_rate_map)

    def __call__(self) -> np.bool:
        """
        Spikes output from each input neuron.

        Returns
        -------
        ndarray[n_inputs, dtype=bool] Spike output for each neuron.
        """
        if not self.values.size:
            return []

        self.network_time += 1
        if self.network_time > self._input_firing_steps:
            return np.zeros(self.values.shape)

        spikes = np.where(
            np.random.uniform(0, 1, size=self.values.size) <= self.values,
            self._magnitude,
            0.0,
        )

        return spikes * self.polarities

    def update(self, state: object):
        """
        Update input generator.

        Parameters
        ----------
        state: object
            Enviornment state in format generator can understand.
        """
        self.network_time = 0 if self._input_firing_steps != -1 else -1000000

        if isinstance(state, (int, float)):
            state = np.array([state])
        else:
            state = np.array(state)

        state = self._state_rate_map[np.int_(state)]

        if not len(state) or self._n_inputs % len(state):
            raise ValueError(f"N_INPUTS must be a multiple of state: {len(state)}!")

        self.values = np.ravel(
            np.array([state for _ in range(self._n_inputs // len(state))])
        )
