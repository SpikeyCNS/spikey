"""
Translator from output neuron spike trains to actions
for the environment.
"""
from spikey.module import Module
import numpy as np


class Readout(Module):
    """
    Translator from output neuron spike trains to actions
    for the environment.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    config = {
        "n_outputs": 10,
        "magnitude": 2,
        "output_range": [-1, 1],
    }
    readout = Readout(**config)

    action = readout(np.ones((10, config["n_outputs"])))
    ```

    ```python
    class network_template(Network):
        config = {
            "n_outputs": 10,
            "magnitude": 2,
            "output_range": [-1, 1],
        }
        _template_parts = {
            "readout": Readout
        }
    ```
    """

    NECESSARY_KEYS = {
        "n_outputs": "int Number of output neurons.",
        "magnitude": "float Spike fire magnitude.",
        "output_range": "list[float] Range of values output can produce.",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, output_spike_train: np.bool) -> object:
        """
        Interpret the output neuron's spike train.

        Parameters
        ----------
        output_spike_train: np.ndarray[t, n_neurons, dtype=bool]
            Spike train with train[-1] being the most recent time.

        Returns
        -------
        object Action chosen.
        """
        raise NotImplementedError(f"__call__ not implemented for {type(self)}!")
