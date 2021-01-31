"""
Spike based stimulus encoding.
"""
import numpy as np


class Input:
    """
    Spike based stimulus encoding.

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
    }
    input = Input(**config)
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
        }
        _template_parts = {
            "inputs": Input
        }
    ```
    """

    NECESSARY_KEYS = {
        "n_inputs": "int Number of inputs.",
        "magnitude": "float Multiplier to each 0, 1 spike value.",
        "firing_steps": "int Number of network steps to fire for, -1 if all.",
        "input_pct_inhibitory": "float Pct of inputs that are inhibitory",
    }

    def __init__(self, **kwargs):
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

        self.polarities = np.where(
            np.random.uniform(0, 1, self._n_inputs) > self._input_pct_inhibitory, 1, -1
        )

        self.values = self.network_time = None

    def __len__(self) -> int:
        """
        Size of input generator == number inputs.
        """
        return self._n_inputs

    def __call__(self) -> np.bool:
        """
        Spikes output from each input neuron.

        Returns
        -------
        ndarray[n_inputs, bool] Spike output for each neuron.
        """
        raise NotImplementedError("Input gen __call__ function not implemented!")

    def update(self, state: object):
        """
        Update input generator.

        Parameters
        ----------
        state: object
            Enviornment state in format generator can understand.
        """
        self.network_time = 0 if self._firing_steps != -1 else -1000000

        self.values = state
