"""
Florian STDP implementation.
Taken from BindsNET.
"""
import numpy as np
from numpy import ndarray
from spikey.module import Module, Key

from spikey.snn.synapse.template import RLSynapse


class FlorianSTDP(RLSynapse):
    """
    Equivalent to MSTDP from BindsNET
    # NOTE This is meant to work on single layers of connections
    """
    NECESSARY_KEYS = RLSynapse.extend_keys(
        [
            Key(
                "processing_time",
                "Number of network timesteps per game timestep.",
                int,
            ),
        ]
    )

    def reset(self):
        """
        Reset Synapse member variables.
        Called at the start of each episode.
        """
        self.trace = np.zeros(self.weights.matrix.shape)
        self.eligibility = np.zeros(self.weights.matrix.shape)

        self.p_plus = np.zeros(shape=self._n_inputs + self._n_neurons)
        self.p_minus = np.zeros(shape=self._n_neurons)

    def _apply_stdp(self, full_spike_log: np.bool, inhibitories: np.int):
        """
        Update synaptic weights via STDP rule.

        Parameters
        ----------
        spike_log: np.array(time, neurons), 0 or 1
            A history of neuron firings with spike_log[-1] is most recent.
        inhibitories: list[int], -1 or 1
            Neuron polarities.
        """
        if not full_spike_log.size:
            return

        try:
            self.spike_log = full_spike_log[-self._stdp_window :]
        except IndexError:
            self.spike_log = full_spike_log

    def reward(self, rwd: float):
        """
        Give synapses a reward.
        Called once per game or network step based on network chosen.

        Parameters
        ----------
        rwd: float
            Reward the network has earned.
        """
        if not hasattr(self, "spike_log"):
            return

        self.tc_plus = 20.0  # Time constant for pre-synaptic firing trace.
        self.tc_minus = 20.0  # Time constant for post-synaptic firing trace.
        self.tc_e_trace = 25.0  # Time constant for the eligibility trace.
        a_plus = 1.0  # Learning rate (post-synaptic).
        a_minus = -1.0  # Learning rate (pre-synaptic).

        # Reshape pre- and post-synaptic spikes.
        source_s = self.spike_log
        target_s = self.spike_log[:, self._n_inputs:]

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.trace *= np.exp(-self._processing_time / self.tc_e_trace)
        self.trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        self.weights._matrix += (
            self._learning_rate * self._processing_time * rwd * self.trace
        )

        # Update P^+ and P^- values.
        self.p_plus *= np.exp(-self._processing_time / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= np.exp(-self._processing_time / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = np.outer(self.p_plus, target_s) + np.outer(
            source_s, self.p_minus
        )

        self.weights.clip()
