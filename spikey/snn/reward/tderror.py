"""
TD Error.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.reward.template import Reward


class TDError(Reward):
    """
    TD Error.
    """

    NECESSARY_KEYS = deepcopy(Reward.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "expected_value": "func Expected value for state.",
            "value_base": "float V_0",
            "value_scale": "float v",
            "n_neurons": "int Number of neurons.",
            "n_outputs": "int Number of output neurons.",
            "processing_time": "int Time network takes to process input.",
            "Tau_r": "int Reward discount constant.",
            "Tau_k": "int Decay time for p(x) kernel.",
            "V_k": "int Rise time for p(x) kernel",
        }
    )

    def __init__(self, **config):
        super().__init__(**config)

        self.time = 0

        self.prev_td, self.prev_value, self.prev_reward = None, None, None

    def __call__(self, state, action):
        # critic_spikes = np.where(self.critic_spikes, 1, 0)
        critic_spikes = np.where(action, 1, 0)

        expected = self._expected_value(state, action, self.time)

        V_0 = self._value_base
        v = self._value_scale

        N = self._n_neurons - self._n_outputs
        processing_time = self._processing_time
        Tau_r = self._Tau_r
        Tau_k = self._Tau_k
        V_k = self._V_k

        times = np.arange(processing_time)[::-1].reshape((-1, 1))

        K = lambda t: (np.exp(-t / Tau_k) - np.exp(-t / V_k)) / (Tau_k - V_k)
        K_dot = lambda t: ((np.exp(-t / V_k) / V_k) - (np.exp(-t / Tau_k) / Tau_k)) / (
            Tau_k - V_k
        )

        K_final = lambda t: K_dot(t) - K(t) / Tau_r
        kernel = K_final(times)

        value_for_td = v / N * np.sum(critic_spikes * kernel)

        value = v / N * np.sum(critic_spikes * K(times)) + V_0

        td = value_for_td - (V_0 / Tau_r) + expected

        if self.time % 1000 == 0:
            p_t = np.mean(critic_spikes)
            p = np.sum(critic_spikes * K(times)) / Tau_r
            p_dot = np.sum(critic_spikes * K_dot(times))
            print(
                f"{self.time:2} | p_t:{p_t:.2f} p:{p:.4f} p':{p_dot:.4f} v0:{V_0 / Tau_r} r:{expected:.4f} td:{td:.4f}"
            )

        self.time += 1

        self.prev_td, self.prev_value, self.prev_reward = td, value, expected

        return td
