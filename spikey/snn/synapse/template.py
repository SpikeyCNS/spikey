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
"""
from spikey.module import Module
import numpy as np


class Synapse(Module):
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
        "inh_weight_mask": None,
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
    synapse = Synapse(w, **config)

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
            "synapses": Synapse
        }
    ```
    """

    NECESSARY_KEYS = {
        "n_neurons": "int Number of neurons in network",
        "n_inputs": "int Number of inputs",
        "stdp_window": "int Time period that stdp will take effect.",
        "learning_rate": "float Scalar to trace updates.",
        "max_weight": "float Max synapse weight.",
        "trace_decay": "float Percent to decay trace by per timestep.",
    }

    def __init__(self, w: object, **kwargs):
        super().__init__(**kwargs)

        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

        self.weights = w

        self.trace = None

    def reset(self):
        """
        Reset Synapse member variables.
        """
        self.trace = np.zeros(
            shape=(self._n_inputs + self._n_neurons, self._n_inputs + self._n_neurons),
            dtype=np.float32,
        )

    def _decay_trace(self):
        """
        Decay eligability trace.
        """
        ## Pre-computing ssaves a considerable amount of time!
        mul = 1 - self._trace_decay

        self.trace *= mul

    def _apply_stdp(self, spike_log: np.bool, inhibitories: np.bool):
        """
        Update synaptic weights via STDP rule.

        Parameters
        ----------
        spike_log: np.array(time, neurons), 0 or 1
            A history of neuron firings with spike_log[-1] is most recent.
        inhibitories: list[int], -1 or 1
            Neuron polarities.
        """
        raise NotImplementedError("Update trace function needs to be implemented!")

    def update(self, spike_log: np.bool, inhibitories: np.int) -> None:
        """
        Update trace for one time step based on decay rule and STDP suggestions.

        Parameters
        ----------
        spike_log: np.array(time, neurons)
            A history of when neurons have spiked, 1 at spike, 0 quiescent with spike_log[-1] is most recent.
        inhibitories: np.array(neurons)
                The polarity, 1 or -1, of each nueron
        """
        ## Decay trace
        self._decay_trace()

        ## Update trace based on stdp suggestions
        self._apply_stdp(spike_log, inhibitories)


class RLSynapse(Synapse):
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

    RLSynapse allows the synapse to be rewarded in order to achieve Reward modulated/learning
    STDP(RMSTDP / RLSTDP). RMSTDP is achieved when a decaying eligability trace of
    reward earned is tracked and used as a factor of the weight update, eg,
    `w += learning_rate * trace * STDP`

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
        "inh_weight_mask": None,
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
    synapse = RLSynapse(w, **config)

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
            "synapses": RLSynapse
        }
    ```
    """

    def reward(self, rwd: float):
        """
        Give synapses a reward.

        Parameters
        ----------
        rwd: float
            Reward the network has earned.
        """
        raise NotImplementedError(f"{type(this)}.reward() has not been implemented!")
