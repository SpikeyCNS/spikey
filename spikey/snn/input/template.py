"""
Spike based stimulus encoding.
"""
import numpy as np
from spikey.module import Module, Key


class Input(Module):
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
        "input_firing_steps": -1,
        "input_pct_inhibitory": 0.2,
    }
    input = Input(**config)
    input.reset()
    env = Logic(preset='XOR')

    state = env.reset()
    for step in range(10):
        input.update(state)

        for _ in range(processing_time)
            in_fires = input()

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
        }
        parts = {
            "inputs": Input
        }
    ```
    """

    NECESSARY_KEYS = [
        Key("n_inputs", "Number of inputs.", int),
        Key("magnitude", "Multiplier to each 0, 1 spike value.", float),
        Key(
            "input_firing_steps",
            "Number of network steps to fire for, -1 if all.",
            int,
            default=-1,
        ),
        Key("input_pct_inhibitory", "Pct of inputs that are inhibitory", float),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self.network_time += 1
        raise NotImplementedError("Input gen __call__ function not implemented!")

    def reset(self):
        """
        Reset Input.
        """

    def update(self, state: object):
        """
        Update input generator.

        Parameters
        ----------
        state: object
            Enviornment state in format generator can understand.
        """
        self.network_time = 0

        try:
            self.values = tuple(state)
        except TypeError:
            self.values = state
