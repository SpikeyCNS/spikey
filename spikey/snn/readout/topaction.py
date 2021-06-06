"""
Return index of highest firing action group, or randomly select from equally highest firing.
"""
import numpy as np

from spikey.module import Key
from spikey.snn.readout.population_vector import PopulationVector


class TopAction(PopulationVector):
    """
    Return index of highest firing action group, or randomly select from equally highest firing.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        config = {
            "n_outputs": 10,
            "magnitude": 2,
            "n_actions": 2,
        }
        readout = TopAction(**config)
        readout.reset()

        action = readout(np.ones((10, config["n_outputs"])))

    .. code-block:: python

        class network_template(Network):
            keys = {
                "n_outputs": 10,
                "magnitude": 2,
                "n_actions": 2,
            }
            parts = {
                "readout": TopAction
            }
    """

    def __call__(self, output_spike_train: np.bool) -> np.float:
        """
        Return index of highest firing action group, or randomly select from equally highest firing.

        Parameters
        ----------
        output_spike_train: np.ndarray[t, n_neurons, dtype=bool]
            Spike train with train[-1] being the most recent time.

        Returns
        -------
        int Index of highest firing action.
        """
        population_vector = super().__call__(output_spike_train)

        maxima = np.max(population_vector)
        idx = np.where(population_vector == maxima)[0]
        action = np.random.choice(idx)

        return action
