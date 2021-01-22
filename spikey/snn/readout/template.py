"""
Template for population readouts.

Override
"""
import numpy as np


class Readout:
    """
    Population readout.

    Parameters
    ----------
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    NECESSARY_KEYS = {
        "n_outputs": "int Number of output neurons.",
        "magnitude": "float Spike fire magnitude.",
        "output_range": "list[float] Range of values output can produce.",
    }

    def __init__(self, **kwargs):
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

    def __call__(self, output_spike_train: np.float) -> object:
        """
        Interpret the population's spikes.

        Parameters
        ----------
        output_spike_train: ndarray [t, n_neurons]
            Spike train, [-1] is most recent time.

        Returns
        -------
        object Chosen action.
        """
        raise NotImplementedError(f"__call__ not implemented for {type(self)}!")
