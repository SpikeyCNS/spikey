"""
Translator from output neuron spike trains to actions
for the environment. Actions set are neuron firing rates.
"""
import numpy as np

from spikey.snn.readout.template import Readout


class NeuronRates(Readout):
    """
    Translator from output neuron spike trains to actions
    for the environment. Actions set are neuron firing rates.

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
        "n_pools": 1,
    }
    readout = NeuronRates(**config)
    readout.reset()

    action = readout(np.ones((10, config["n_outputs"])))
    ```

    ```python
    class network_template(Network):
        config = {
            "n_outputs": 10,
            "magnitude": 2,
            "output_range": [-1, 1],
            "n_pools": 1,
        }
        _template_parts = {
            "readout": NeuronRates
        }
    ```
    """

    NECESSARY_KEYS = Readout.extend_keys(
        {
            "n_pools": "int Number of groups to put neurons into. 0 pools means each neuron separate output.",
        }
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._n_pools == 0:
            self._n_pools = self._n_outputs

    def __call__(self, output_spike_train: np.bool) -> np.float:
        """
        Interpret the output neuron's spike train into pool firing rates.

        Parameters
        ----------
        output_spike_train: np.ndarray[t, n_neurons, dtype=bool]
            Spike train with train[-1] being the most recent time.

        Returns
        -------
        ndarray[n_pools, dtype=float] Firing rate of each neuron pool.
        """
        if self._n_outputs == 0:
            return 0

        idx = np.linspace(0, self._n_outputs, self._n_pools + 1).astype(np.int)
        pools = [output_spike_train[idx[i] : idx[i + 1]] for i in range(self._n_pools)]
        return np.mean(pools, axis=(1, 2))
