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


class LTPET(RLSynapse):
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

    Examples
    --------

    .. code-block:: python

        w_config = {
            "n_neurons": 50,
            "n_inputs": 0,
            "matrix": np.random.uniform(size=(10, 10)),
        }
        w = Manual(**config)

        config = {
            "n_neurons": 50,
            "n_inputs": 0,
            "stdp_window": 200,
            "learning_rate": .05,
            "trace_decay": .1,
        }
        synapse = LTPET(w, **config)
        synapse.reset()

        pre_fires = np.random.uniform(size=config['n_neurons']) <= .08
        post_fires = np.matmul(w.matrix, pre_fires) >= 2
        spike_log = np.vstack((post_fires, pre_fires))
        synapse.update(spike_log, np.zeros(config['n_neurons]))

    .. code-block:: python

        class network_template(Network):
            keys = {
                "n_neurons": 50,
                "n_inputs": 10,
                "stdp_window": 200,
                "learning_rate": .05,
                "trace_decay": .1,
            }
            parts = {
                "synapses": LTPET
            }
    """

    def reset(self):
        """
        Reset Synapse member variables.
        Called at the start of each episode.
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
        if not full_spike_log.size:
            return

        try:
            spike_log = full_spike_log[-self._stdp_window :]
        except IndexError:
            spike_log = full_spike_log

        pre_locs = np.where(np.any(spike_log[:-1], axis=0))[0]
        post_locs = np.where(spike_log[-1])[0]

        if not pre_locs.size or not post_locs.size:
            return

        decay_multiplier = np.arange(1, spike_log.shape[0]).reshape((-1, 1))
        dts = decay_multiplier * spike_log[:-1]
        dts = np.sum(dts, axis=0)

        update_mult = self._learning_rate / self._stdp_window * self.trace

        self._hebbian(pre_locs, post_locs, inhibitories, dts, update_mult)

        self.weights.clip()

    def reward(self, rwd: float):
        """
        Give synapses a reward.
        Called once per game or network step based on network chosen.

        Parameters
        ----------
        rwd: float
            Reward the network has earned.
        """
        # self.trace += rwd
        self.trace = rwd
