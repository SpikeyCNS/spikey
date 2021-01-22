"""
Template for neurons that spike based on a pre-computed spike schedule.
"""
import numpy as np


class Neuron:
    """
    A group of spiking neurons, fire according to a pre-computed spike schedule.
    """

    NECESSARY_KEYS = {
        "magnitude": "float Magnitude of spike.",
        "n_neurons": "int Number of neurons in the network.",
        "neuron_pct_inhibitory": "float [0, 1] Percentage of inhibitory neurons.",
        "potential_decay": "float[0, 1] Percentage voltage loss on each tick.",
        "prob_rand_fire": "float [0, 1] Probability each neuron will randomly fire",
        "refractory_period": "int Amount of time after spike neuron cannot fire.",
        "resting_mv": "float Neuron resting voltage.",
        "spike_delay": "int[0, 10] Units of time after hitting threshold to fire.",
    }

    def __init__(self, **kwargs):
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

        polarities = np.random.uniform(size=self._n_neurons)
        self.polarities = np.where(polarities < self._neuron_pct_inhibitory, -1.0, 1.0)

        if "polarities" in kwargs:
            self.polarities = np.array(kwargs["polarities"])

        ## Initialized in self.reset()
        self.potentials = self.refractory_timers = None
        self.spike_shape = self.schedule = None

    def reset(self):
        """
        Reset neuron.
        """
        self.potentials = self._resting_mv * np.ones(self._n_neurons, dtype="float16")

        self.refractory_timers = np.zeros(self._n_neurons)

        self.spike_shape = self._generate_spike_shape()
        self.schedule = np.zeros(shape=(self.spike_shape.size, self._n_neurons))

    def _generate_spike_shape(self) -> np.bool:
        """
        Called once per reset, generates the array that will be
        added to the schedule.

        Returns
        -------
        Boolean np.array width=1, variable height.
        """
        SCHEDULE_LENGTH = 10
        spike_shape = np.zeros(shape=(SCHEDULE_LENGTH, 1))

        spike_shape[self._spike_delay] = 1

        return spike_shape

    def __ge__(self, threshold: float) -> np.bool:
        """
        Schedule spikes for neurons above threshold, spike based on schedule.

        Parameters
        ----------
        threshold: float
            Spiking threshold, neurons schedule spikes if potentials >= threshold.

        Returns
        -------
        Neuron outputs.
        """
        spike_occurences = self.potentials >= threshold

        ## Leaky
        spike_occurences += (
            np.random.uniform(0, 1, size=self._n_neurons) < self._prob_rand_fire
        )

        spike_occurences &= self.refractory_timers <= 0

        self.refractory_timers[spike_occurences] = self._refractory_period + 1
        self.schedule += self.spike_shape * np.int_(spike_occurences)

        output = self.schedule[0] * self.polarities * self._magnitude

        return output

    def __iadd__(self, incoming_v: np.float):
        """
        Add to membrane potentials.

        Parameters
        ----------
        incoming_v: np.array(neurons, dtype=float)
            Amount to increase neuron potentials by.
        """
        self.potentials += incoming_v

        return self

    def update(self):
        """
        Update neuron voltage and handle refractory dynamics.
        """
        self.refractory_timers -= 1

        self.schedule = np.vstack((self.schedule[1:], np.zeros(shape=self._n_neurons)))

        self.potentials[
            np.where(self.refractory_timers > 0)
        ] = -65499.0  # finfo('float16').min
        self.potentials[np.where(self.refractory_timers == 0)] = self._resting_mv

        decay = 1 - self._potential_decay
        self.potentials = (
            self.potentials - self._resting_mv
        ) * decay + self._resting_mv
