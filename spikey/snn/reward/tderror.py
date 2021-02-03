"""
Temporal difference residual function.
TD(t) = -V'(state_t) - 1/Tau_r * V(state_t) + r(state_t, action_t)
V & V' calculated with kernel K on the critic spike train, r=expected_value.
K = (exp(-t / Tau_k) - exp(-t - V_k)) / (Tau_k - V_k)

Frémaux N, Sprekeler H, Gerstner W (2013) Reinforcement Learning Using a
Continuous Time Actor-Critic Framework with Spiking Neurons. PLOS
Computational Biology 9(4): e1003024. https://doi.org/10.1371/journal.pcbi.1003024

https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Fremaux13.pdf
"""
from copy import deepcopy
import numpy as np

from spikey.snn.reward.template import Reward


class TDError(Reward):
    """
    Temporal difference residual function.
    TD(t) = -V'(state_t) - 1/Tau_r * V(state_t) + r(state_t, action_t)
    V & V' calculated with kernel K on the critic spike train, r=expected_value.
    K = (exp(-t / Tau_k) - exp(-t - V_k)) / (Tau_k - V_k)

    Frémaux N, Sprekeler H, Gerstner W (2013) Reinforcement Learning Using a
    Continuous Time Actor-Critic Framework with Spiking Neurons. PLOS
    Computational Biology 9(4): e1003024. https://doi.org/10.1371/journal.pcbi.1003024

    https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Fremaux13.pdf

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    config = {
        "reward_mult": 1,
        "punish_mult": -2,
    }
    rewarder = Reward(**config)
    rewarder.reset()

    r = rewarder(state, action)
    ```

    ```python
    class network_template(Network):
        config = {
            "reward_mult": 1,
            "punish_mult": -2,
        }
        _template_parts = {
            "rewarder": Reward
        }
    ```
    """

    NECESSARY_KEYS = deepcopy(Reward.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "processing_time": "int Number of network timesteps per game timestep.",
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

        self.prev_td, self.prev_value, self.prev_reward = 0, 0, 0

    def reset(self):
        """
        Reset rewarder member variables.
        """
        self.time = 0
        self.prev_td, self.prev_value, self.prev_reward = 0, 0, 0

    def __call__(self, state: object, action: object) -> float:
        """
        Determine how much reward should be given for taking action in state.

        Parameters
        ----------
        state: any
            Environment state before action is taken.
        action: any
            Action taken in response to state.

        Returns
        -------
        float Reward for taking action in state.
        """
        if self.time < self._processing_time:
            self.prev_td, self.prev_value, self.prev_reward = 0, 0, 0
            self.time += 1
            return 0

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
