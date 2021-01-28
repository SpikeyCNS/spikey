"""
A group of spiking neurons.

Each spiking neuron has an internal membrane potential that
increases with each incoming spike. The potential persists but slowly
decreases over time. Each neuron fires when its potential surpasses
some firing threshold and does not fire again for the duration
of its refractory period.
"""
import numpy as np


class Neuron:
    """
    A group of spiking neurons.

    Each spiking neuron has an internal membrane potential that
    increases with each incoming spike. The potential persists but slowly
    decreases over time. Each neuron fires when its potential surpasses
    some firing threshold and does not fire again for the duration
    of its refractory period.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    config = {
        "magnitude": 2,
        "n_neurons": 100,
        "neuron_pct_inhibitory": .2,
        "potential_decay": .2,
        "prob_rand_fire": .08,
        "refractory_period": 1,
        "resting_mv": 0,
        "spike_delay": 0,
    }
    neurons = Neuron(**config)
    neurons.reset()

    weights = np.random.uniform(0, 2, size=(config['n_neurons'], config['n_neurons]))

    for i in range(100):
        spikes = self.neurons >= 16

        self.neurons.update()

        neurons += np.sum(
            weights * spikes.reshape((-1, 1)), axis=0
        )
    ```

    ```python
    class network_template(Network):
        config = {
            "magnitude": 2,
            "n_neurons": 100,
            "neuron_pct_inhibitory": .2,
            "potential_decay": .2,
            "prob_rand_fire": .08,
            "refractory_period": 1,
            "resting_mv": 0,
            "spike_delay": 0,
        }
        _template_parts = {
            "neurons": Neuron
        }
    ```
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
        Reset all neuron members.
        """
        self.potentials = self._resting_mv * np.ones(self._n_neurons, dtype="float16")

        self.refractory_timers = np.zeros(self._n_neurons)

        self.spike_shape = self._generate_spike_shape()
        self.schedule = np.zeros(shape=(self.spike_shape.size, self._n_neurons))

    def _generate_spike_shape(self) -> np.bool:
        """
        Generate neuron output schedule for time after it's potential passes
        the firing threshold.

        Returns
        -------
        ndarray[SCHEDULE_LENGTH, bool] Neuron output schedule.
        """
        SCHEDULE_LENGTH = 10
        spike_shape = np.zeros(shape=(SCHEDULE_LENGTH, 1))

        spike_shape[self._spike_delay] = 1

        return spike_shape

    def __ge__(self, threshold: float) -> np.bool:
        """
        Determine whether each neuron will fire or not according to threshold.

        Parameters
        ----------
        threshold: float
            Spiking threshold, neurons schedule spikes if potentials >= threshold.

        Returns
        -------
        ndarray[n_neurons, bool] Spike output from each neuron at the current timestep.

        Usage
        -----
        ```python
        config = {
            "magnitude": 2,
            "n_neurons": 100,
            "neuron_pct_inhibitory": .2,
            "potential_decay": .2,
            "prob_rand_fire": .08,
            "refractory_period": 1,
            "resting_mv": 0,
            "spike_delay": 0,
        }
        neurons = Neuron(**config)
        neurons.reset()

        weights = np.random.uniform(0, 2, size=(config['n_neurons'], config['n_neurons]))

        for i in range(100):
            spikes = self.neurons >= 16

            self.neurons.update()

            neurons += np.sum(
                weights * spikes.reshape((-1, 1)), axis=0
            )
        ```
        """
        spike_occurences = self.potentials >= threshold

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
        Add incoming voltage to the neurons membrane potentials.

        Parameters
        ----------
        incoming_v: np.ndarray[neurons, dtype=float]
            Amount to increase each neuron's potential by.

        Usage
        -----
        ```python
        config = {
            "magnitude": 2,
            "n_neurons": 100,
            "neuron_pct_inhibitory": .2,
            "potential_decay": .2,
            "prob_rand_fire": .08,
            "refractory_period": 1,
            "resting_mv": 0,
            "spike_delay": 0,
        }
        neurons = Neuron(**config)
        neurons.reset()

        weights = np.random.uniform(0, 2, size=(config['n_neurons'], config['n_neurons]))

        for i in range(100):
            spikes = self.neurons >= 16

            self.neurons.update()

            neurons += np.sum(
                weights * spikes.reshape((-1, 1)), axis=0
            )
        ```
        """
        self.potentials += incoming_v

        return self

    def update(self):
        """
        Simulate the neurons for one time step. Update membrane potentials
        and manage refractory dynamics.

        Usage
        -----
        ```python
        config = {
            "magnitude": 2,
            "n_neurons": 100,
            "neuron_pct_inhibitory": .2,
            "potential_decay": .2,
            "prob_rand_fire": .08,
            "refractory_period": 1,
            "resting_mv": 0,
            "spike_delay": 0,
        }
        neurons = Neuron(**config)
        neurons.reset()

        weights = np.random.uniform(0, 2, size=(config['n_neurons'], config['n_neurons]))

        for i in range(100):
            spikes = self.neurons >= 16

            self.neurons.update()

            neurons += np.sum(
                weights * spikes.reshape((-1, 1)), axis=0
            )
        ```
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
