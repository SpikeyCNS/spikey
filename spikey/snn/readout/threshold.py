"""
Translator from output neuron spike trains to actions
for the environment. Actioned determined based on neuron
firing rate greater than action_threshold or not, as
`output_range[firing_rate >= action_threshold]`.
"""
import numpy as np

from spikey.module import Module, Key
from spikey.snn.readout.template import Readout


class Threshold(Readout):
    """
    Translator from output neuron spike trains to actions
    for the environment. Actioned determined based on neuron
    firing rate greater than action_threshold or not, as
    `output_range[firing_rate >= action_threshold]`.

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
        "action_threshold": .5,
    }
    readout = Threshold(**config)
    readout.reset()

    action = readout(np.ones((10, config["n_outputs"])))
    ```

    ```python
    class network_template(Network):
        config = {
            "n_outputs": 10,
            "magnitude": 2,
            "output_range": [-1, 1],
            "action_threshold": .5,
        }
        parts = {
            "readout": Threshold
        }
    ```
    """

    NECESSARY_KEYS = Readout.extend_keys(
        [
            Key(
                "action_threshold",
                "float or 'mean' Output neuron rate threshold to trigger high state.",
            ),
            Key(
                "output_range",
                "list[float] Range of values output can produce.",
                default=[0, 1],
            ),
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._action_threshold == "mean":
            self.rate_log = []

    def __call__(self, output_spike_train: np.bool) -> object:
        """
        Interpret the output neuron's spike train.

        Parameters
        ----------
        output_spike_train: np.ndarray[t, n_neurons, dtype=bool]
            Spike train with train[-1] being the most recent time.

        Returns
        -------
        output_range[rate >= threshold] Selected action based on whether rate was
        greater than threshold or not.
        """
        if self._n_outputs == 0:
            return 0

        rate = np.mean(output_spike_train) / self._magnitude
        if self._action_threshold == "mean":
            threshold = np.mean(self.rate_log) if self.rate_log else 0
            self.rate_log.append(rate)
        else:
            threshold = self._action_threshold
        action = self._output_range[bool(rate >= threshold)]

        return action
