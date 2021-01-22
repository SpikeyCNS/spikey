"""
Template for synapses.

Override
"""
import numpy as np


class Synapse:
    """
    Hedonistic synapses updating weights based on stdp suggestions.

    Parameters
    ----------
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    NECESSARY_KEYS = {
        "n_neurons": "int Number of neurons in network",
        "n_inputs": "int Number of inputs",
        "stdp_window": "int Time period that stdp will take effect.",
        "learning_rate": "float Scalar to trace updates.",
        "max_weight": "float Max synapse weight.",
        "trace_decay": "float Percent to decay trace by per timestep.",
    }

    def __init__(self, w: "Weight", **kwargs):
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

        self.weights = w

        ## Initialized in self.reset()
        self.trace = None

    def reset(self):
        """
        Reset synapses.
        """
        self.trace = np.zeros(
            shape=(self._n_inputs + self._n_neurons, self._n_inputs + self._n_neurons),
            dtype=np.float32,
        )

    def _decay_trace(self):
        """
        Decay a single trace value.

        Effects
        -------
        Decayed trace value.
        """
        ## Pre-computing ssaves a considerable amount of time!
        mul = 1 - self._trace_decay

        self.trace *= mul

    def _apply_stdp(self, spike_log: np.bool, inhibitories: np.bool):
        """
        Use stdp to update trace based on dt.

        Parameters
        ----------
        spike_log: np.array(time, neurons), 0 or 1
            A history of neuron firings.
        inhibitories: list[int], -1 or 1
            Neuron polarities.

        Returns
        -------
        Trace value with stdp suggestion.
        """
        raise NotImplementedError("Update trace function needs to be implemented!")

    def update(self, spike_log: np.bool, inhibitories: np.bool) -> None:
        """
        Update trace based on stdp suggestions.

        Parameters
        ----------
        spike_log: np.array(time, neurons)
            A history of when neurons have spiked, 1 at spike, 0 quiescent.
        inhibitories: np.array(neurons)
                The polarity, 1 or -1, of each nueron
        """
        ## Decay trace
        self._decay_trace()

        ## Update trace based on stdp suggestions
        self._apply_stdp(spike_log, inhibitories)


class RLSynapse(Synapse):
    """
    Hedonistic synapses updating weights based on stdp suggestions and reward.

    Parameters
    ----------
    n_inputs: int
        Number of input streams.
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    def reward(self, rwd: float):
        """
        Update weights based on trace and reward.

        Parameters
        ----------
        rwd: float
            Reward the network has earned.
        """
        self.weights += self.trace[:, self._n_inputs :] * rwd
