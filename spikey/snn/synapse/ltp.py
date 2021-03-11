"""
Hedonistic synapses updating weights based on stdp suggestions.
The weight matrix defines how much charge from pre-synaptic neurons
goes to which post-synaptic neurons. The weight matrix is stored in
and managed by the Weight class, stored in Synapse as self.weight.
Synapse defines the learning behavior of the synapses(weights) of
the network based on neuron spike times.

The Spike-timing-dependent synaptic plasticity(STDP) learning algorithm is
a variant of the fire together wire together rule. Similar to hebbian learning,
for any synapse, if the pre-synaptic neuron tends to fire soon before the
post-synaptic neuron, the synapses weight will increase. If the opposite
tends to happen, post before pre firings, the weight will decrease. Often times
the eligability trace of some sparse variable(eg dopaime reward) is tracked and
is used as a factor of the update rule along with learning rate.
LTP is the first part of STDP, the fire together wire together part where LTP
is what is defined in this class.

RLSynapse allows the synapse to be rewarded in order to achieve Reward modulated/learning
STDP(RMSTDP / RLSTDP). RMSTDP is achieved when a decaying eligability trace of
reward earned is tracked and used as a factor of the weight update, eg,
`w += learning_rate * trace * LTP`
"""
import numpy as np

from spikey.snn.synapse.template import RLSynapse


class LTP(RLSynapse):
    """
    Hedonistic synapses updating weights based on stdp suggestions.
    The weight matrix defines how much charge from pre-synaptic neurons
    goes to which post-synaptic neurons. The weight matrix is stored in
    and managed by the Weight class, stored in Synapse as self.weight.
    Synapse defines the learning behavior of the synapses(weights) of
    the network based on neuron spike times.

    The Spike-timing-dependent synaptic plasticity(STDP) learning algorithm is
    a variant of the fire together wire together rule. Similar to hebbian learning,
    for any synapse, if the pre-synaptic neuron tends to fire soon before the
    post-synaptic neuron, the synapses weight will increase. If the opposite
    tends to happen, post before pre firings, the weight will decrease. Often times
    the eligability trace of some sparse variable(eg dopaime reward) is tracked and
    is used as a factor of the update rule along with learning rate.
    LTP is the first part of STDP, the fire together wire together part where LTP
    is what is defined in this class.

    RLSynapse allows the synapse to be rewarded in order to achieve Reward modulated/learning
    STDP(RMSTDP / RLSTDP). RMSTDP is achieved when a decaying eligability trace of
    reward earned is tracked and used as a factor of the weight update, eg,
    `w += learning_rate * trace * LTP`

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    w_config = {
        "n_neurons": 50,
        "n_inputs": 0,
        "max_weight": 3,
        "matrix": np.random.uniform(size=(10, 10)),
    }
    w = Manual(**config)

    config = {
        "n_neurons": 50,
        "n_inputs": 0,
        "max_weight": 3,
        "stdp_window": 200,
        "learning_rate": .05,
        "trace_decay": .1,
    }
    synapse = LTP(w, **config)
    synapse.reset()

    pre_fires = np.random.uniform(size=config['n_neurons']) <= .08
    post_fires = np.matmul(w.matrix, pre_fires) >= 2
    spike_log = np.vstack((post_fires, pre_fires))
    synapse.update(spike_log, np.zeros(config['n_neurons]))
    ```

    ```python
    class network_template(Network):
        config = {
            "n_neurons": 50,
            "n_inputs": 10,
            "stdp_window": 200,
            "learning_rate": .05,
            "max_weight": 3,
            "trace_decay": .1,
        }
        _template_parts = {
            "synapses": LTP
        }
    ```
    """

    def reset(self):
        """
        Reset Synapse member variables.
        """
        self.trace = 0

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
        ## Find how long ago each neuron fired.
        try:
            spike_log = full_spike_log[-self._stdp_window - 1 :]
        except IndexError:
            spike_log = full_spike_log

        pre_locations = np.where(np.any(spike_log[:-1], axis=0))[0]
        post_locations = np.where(spike_log[-1])[0]

        if not pre_locations.size or not post_locations.size:
            return

        max_time_diff = min(self._stdp_window, spike_log.shape[0])
        decay_multiplier = np.arange(max_time_diff - 1, -1, -1).reshape((-1, 1))
        decayed_fires = decay_multiplier * spike_log
        dts = np.where(decayed_fires, self._stdp_window + 1 - decayed_fires, 0)
        dts = np.sum(dts, axis=0)

        update_mult = self._learning_rate / self._stdp_window * self.trace

        self._hebbian(pre_locs, post_locs, inhibitories, dts, update_mult)

        np.clip(
            self.weights._matrix.data,
            0.0,
            float(self.weights._max_weight),
            out=self.weights._matrix.data,
        )

    def reward(self, rwd: float):
        """
        Give synapses a reward.

        Parameters
        ----------
        rwd: float
            Reward the network has earned.
        """
        # self.trace += rwd
        self.trace = rwd
