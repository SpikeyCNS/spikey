"""
Population vector coding readout from output neuron spike trains to actions
for the environment.
"""
import numpy as np

from spikey.module import Key
from spikey.snn.readout.template import Readout


class PopulationVector(Readout):
    """
    Population vector coding readout from output neuron spike trains to actions
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
        "n_actions": 2,
    }
    readout = PopulationVector(**config)
    readout.reset()

    action = readout(np.ones((10, config["n_outputs"])))
    ```

    ```python
    class network_template(Network):
        keys = {
            "n_outputs": 10,
            "magnitude": 2,
            "n_actions": 2,
        }
        parts = {
            "readout": PopulationVector
        }
    ```
    """

    NECESSARY_KEYS = Readout.extend_keys(
        [
            Key("n_actions", "Number of action groups.", int),
        ]
    )

    def __call__(self, output_spike_train: np.bool) -> np.float:
        """
        Interpret the output neuron's spike train via population vector coding.

        Parameters
        ----------
        output_spike_train: np.ndarray[t, n_neurons, dtype=bool]
            Spike train with train[-1] being the most recent time.

        Returns
        -------
        ndarray[n_actions, dtype=float] Normalized rate from each output pool.
        """
        if self._n_outputs == 0:
            return np.zeros(self._n_actions)

        spikes = np.where(output_spike_train, 1, 0)

        T_gamma, V_gamma = 50, 20
        gamma = lambda t: (np.e ** (-t / T_gamma) - np.e ** (-t / V_gamma)) / (
            T_gamma - V_gamma
        )
        decay_kernel = np.vectorize(gamma)(np.arange(spikes.shape[0])).reshape((-1, 1))

        firing_rates = np.sum(spikes * decay_kernel, axis=0)

        group_size = self._n_outputs // self._n_actions

        p = [
            np.sum(firing_rates[i * group_size : (i + 1) * group_size])
            for i in range(self._n_actions)
        ]
        p = np.array(p)

        if np.sum(p) != 0:
            actions = p / np.sum(p)
        else:
            actions = np.ones(p.shape) / p.size

        return actions
