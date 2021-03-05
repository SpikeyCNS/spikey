"""
Translator from output neuron spike trains to actions
for the environment.
"""
import numpy as np
from spikey.module import Module, Key


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
    }
    readout = Readout(**config)
    readout.reset()

    action = readout(np.ones((10, config["n_outputs"])))
    ```

    ```python
    class network_template(Network):
        config = {
            "n_outputs": 10,
            "magnitude": 2,
        }
        _template_parts = {
            "readout": Readout
        }
    ```
    """

    NECESSARY_KEYS = [
        Key("n_outputs", "Number of output neurons.", int),
        Key("magnitude", "Spike fire magnitude.", float),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        """
        Reset all readout members.
        """
        pass

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
