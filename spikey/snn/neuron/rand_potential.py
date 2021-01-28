"""
prob_rand_fire affect potentials directly instead of firing.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.neuron.template import Neuron


class RandPotential(Neuron):
    """
    Neurons where prob_rand_fire rate affects potentials instead of spikes.
    """

    NECESSARY_KEYS = deepcopy(Neuron.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {"leak_scalar": "float Multiplier of leak to add to potential."}
    )

    def __ge__(self, threshold: float) -> np.ndarray:
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
        ## Leaky
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
