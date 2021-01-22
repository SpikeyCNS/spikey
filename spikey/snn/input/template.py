"""
Template for input types.
"""
import numpy as np


class Input:
    """
    Network input template.

    Parameters
    ----------
    n_inputs: int
        Number of inputs.
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

        ## Initialized w/ update
        self.values = self.network_time = None

    def __len__(self) -> int:
        """
        The length of the generator is the number of inputs.
        """
        return self._n_inputs

    def __call__(self) -> np.bool:
        """
        Spike output for each input neuron.

        Output expected to abide by self.polarities!

        Returns
        -------
        ndarray Spike output for each neuron.
        """
        raise NotImplementedError("Input gen __call__ function not implemented!")

    def update(self, state: object):
        """
        Update input settings.

        Parameters
        ----------
        state: list of float
            Discretized enviornment state.
        """
        self.network_time = 0 if self._firing_steps != -1 else -1000000

        self.values = state
