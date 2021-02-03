"""
A group of spiking neurons with noise `~U(0, leak_scalar)` is added
to `n_neurons * prob_rand_fire` neurons at each step.

Each spiking neuron has an internal membrane potential that
increases with each incoming spike. The potential persists but slowly
decreases over time. Each neuron fires when its potential surpasses
some firing threshold and does not fire again for the duration
of its refractory period.
"""
import numpy as np

from spikey.snn.neuron.template import Neuron


class RandPotential(Neuron):
    """
    A group of spiking neurons with noise `~U(0, leak_scalar)` is added
    to `n_neurons * prob_rand_fire` neurons at each step.

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
        "leak_scalar": .1,
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
            "leak_scalar": .1,
        }
        _template_parts = {
            "neurons": Neuron
        }
    ```
    """

    NECESSARY_KEYS = Neuron.extend_keys(
        {"leak_scalar": "float Multiplier of leak to add to potential."}
    )

    def __ge__(self, threshold: float) -> np.bool:
        """
        Add noise `~U(0, leak_scalar)` to `n_neurons * prob_rand_fire` neurons
        then determine whether each neuron will fire or not according to threshold.

        Parameters
        ----------
        threshold: float
            Spiking threshold, neurons schedule spikes if potentials >= threshold.

        Returns
        -------
        ndarray[n_neurons, dtype=bool] Spike output from each neuron at the current timestep.

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
            "leak_scalar": .1,
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
        noise = np.random.uniform(0, self._leak_scalar, size=self._n_neurons)
        noise[
            ~(np.random.uniform(0, 1, size=self._n_neurons) <= self._prob_rand_fire)
        ] = 0

        self.potentials += noise

        spike_occurences = self.potentials >= threshold

        self.refractory_timers[spike_occurences] = self._refractory_period + 1
        self.schedule += self.spike_shape * np.int_(spike_occurences)

        output = self.schedule[0] * self.polarities * self._magnitude

        return output
