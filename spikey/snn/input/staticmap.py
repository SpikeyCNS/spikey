"""
Custom state - input firings mapping.
"""
import numpy as np

from spikey.module import Key
from spikey.snn.input.template import Input


class StaticMap(Input):
    """
    Custom state - input firings mapping.

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
        "firing_steps": -1,
        "input_pct_inhibitory": 0.2,
        "mapping": {
            (1, 0): np.random.uniform(20, 10) <= .8,
            (.5, .5): np.random.uniform(20, 10) <= .3
        },
    }
    input = StaticMap(**config)
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
            "firing_steps": -1,
            "input_pct_inhibitory": 0.2,
            "mapping": {
                'state1': np.random.uniform(20, 10) <= .5,
                'state2': np.random.uniform(20, 10) <= .5
                },
        }
        _template_parts = {
            "inputs": StaticMap
        }
    ```
    """

    NECESSARY_KEYS = Input.extend_keys(
        [
            Key(
                "mapping",
                "dict[tuple]->ndarray[processing_time, n_inputs, dtype=bool] State to fires mapping..",
            )
        ]
    )

    def __call__(self) -> np.bool:
        """
        Spikes output from each input neuron.

        Returns
        -------
        ndarray[n_inputs, dtype=bool] Spike output for each neuron.
        """
        output = np.array(self._mapping[self.values])

        if len(output.shape) > 1:
            spikes = [value * self._magnitude for value in output[self.time]]
        else:
            spikes = [value * self._magnitude for value in output]

        self.time += 1

        return np.array(spikes) * self.polarities

    def update(self, state: object):
        """
        Update input generator.

        Parameters
        ----------
        state: object
            Enviornment state in format generator can understand.
        """
        self.time = 0

        try:
            self.values = tuple(state)
        except TypeError:
            self.values = state
