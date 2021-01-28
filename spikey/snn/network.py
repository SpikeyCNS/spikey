"""
Spiking neural network core driver.
"""
from copy import deepcopy
import numpy as np


class Network:
    """
    Matrix based spiking neural network.

    Parameters
    ----------
    callback:
        ...
    game: RL, default=None
        Game to pull parameters from.

    Parameter Priorities
    --------------------
    Highest. Parameters passed directly
    -------. Network.config
    Lowest . Game parameters.

    Templating
    ----------
    config: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Usage
    -----
    Create an SNN object, run .reset() then .tick(state) repeatedly until end game.
    """

    NECESSARY_KEYS = {
        "n_inputs": "int Number input neurons.",
        "n_outputs": "int n_outputs = n_neurons - n_body.",
        "processing_time": "int Number of network timesteps per game timestep."
        + "NOTE: processing_time must be greater than window!",
        "firing_threshold": "float Neuron voltage threshold to fire.",
        "n_neurons": "int Number of neurons in the network.",
    }
    NECESSARY_PARTS = {
        "inputs": "snn.input.Input",
        "neurons": "snn.neuron.Neuron",
        "weights": "snn.weight.Weight",
        "synapses": "snn.synapse.Synapse",
        "readout": "snn.readout.Readout",
        "modifiers": "list of snn.modifier.Modifier",
    }

    def __init__(
        self,
        callback: object = None,
        game: object = None,
        **kwargs,
    ):
        self.parts = {"modifiers": None}
        self.parts.update(self._template_parts)
        for key in self.NECESSARY_PARTS:
            if key in kwargs:
                self.parts[key] = kwargs[key]

        self._params = deepcopy(game.params) if game is not None else {}
        self._params.update(self.config)
        self._params.update(kwargs)

        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", self._params[key])

        self.callback = (
            callback
            or type(
                "NotCallback",
                (object,),
                {"__getattr__": lambda s, k: lambda *a, **kw: False},
            )()
        )

        self._n_inputs = self._params["n_inputs"]
        self._n_outputs = self._params["n_outputs"]

        ## Network parts
        for name in self.NECESSARY_PARTS.keys():
            part = self.parts[name]

            if name == "synapses":
                value = part(self.weights, **self.params)
            elif part is None:
                value = part
            else:
                value = part(**self.params)

            setattr(self, name, value)

        self.synapses.weights = self.weights

        ## Initialized in self.reset()
        self.internal_time = self._spike_log = None

        self.callback.network_init(self)

    @property
    def params(self) -> dict:
        """
        Read only configuration of network.
        """
        return deepcopy(self._params)

    @property
    def spike_log(self) -> np.bool:
        """
        Spike log of the most recent tick.
        """
        try:
            return self._spike_log[-self._processing_time :]
        except TypeError:
            return None

    @classmethod
    def list_necessary_keys(cls, **kwargs):
        """
        Print list of all required keys for this network configuration.

        Parameters
        ----------
        kwargs: extended parts list
        """
        parts = {}
        parts.update(cls._template_parts)
        for key in cls.NECESSARY_PARTS:
            if key in kwargs:
                parts[key] = kwargs[key]

        for part in [cls] + list(parts.values()):
            if part is None:
                continue

            print(part.__name__)

            for key, value in part.NECESSARY_KEYS.items():
                print(f"\t{key}: {value}")

    def reset(self):
        """
        Set network to initial state.
        """
        self.internal_time = 0

        self.neurons.reset()
        self.synapses.reset()

        self._spike_log = np.zeros(
            (
                self.synapses._stdp_window + self._processing_time,
                self._n_inputs + self._n_neurons,
            ),
            dtype=np.float16,
        )

        self.callback.network_reset()

    def tick(self, state: object) -> object:
        """
        Act based on input data.

        Parameters
        ----------
        state: immutable
            The current game input.

        Returns
        -------
        Corresponding output.
        """
        polarities = np.append(np.ones(self._n_inputs), self.neurons.polarities)

        self._spike_log[: self.synapses._stdp_window] = self._spike_log[
            -self.synapses._stdp_window :
        ]
        normalized_spike_log = self._spike_log.astype(np.bool_)

        self.inputs.update(state)

        if self.modifiers is not None:
            for modifier in self.modifiers:
                modifier.update(self)

        for i in range(self._processing_time):
            self.internal_time += 1

            spikes = np.append(self.inputs(), self.neurons >= self._firing_threshold)

            self._spike_log[self.synapses._stdp_window + i] = spikes
            normalized_spike_log[self.synapses._stdp_window + i] = spikes.astype(
                np.bool_
            )

            self.neurons.update()
            self.synapses.update(
                normalized_spike_log[i : i + self.synapses._stdp_window], polarities
            )

            self.neurons += np.sum(
                self.synapses.weights * spikes.reshape((-1, 1)), axis=0
            )

        outputs = self._spike_log[-self._processing_time :, -self._n_outputs :][::-1]
        output = self.readout(outputs)

        self.callback.network_tick(state, output)
        return output


class RLNetwork(Network):
    """
    Matrix based spiking neural network with functions build for RL.

    Parameters
    ----------
    callback:
        ...
    game: RL, default=None
        Game to pull parameters from.

    Parameter Priorities
    --------------------
    Highest. Parameters passed directly
    -------. Network.config
    Lowest . Game parameters.

    Templating
    ----------
    config: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Usage
    -----
    Create an SNN object, run .reset() then .tick(state) repeatedly until end game.
    """

    NECESSARY_PARTS = deepcopy(Network.NECESSARY_PARTS)
    NECESSARY_PARTS.update(
        {
            "rewarder": "snn.reward.Reward",
        }
    )

    def __init__(
        self,
        callback: object = None,
        game: object = None,
        **kwargs,
    ):
        super().__init__(callback=callback, game=game, **kwargs)

    def reward(self, state: object, action: object, reward: float = None) -> float:
        """
        Calculate reward and apply to synapses.
        """
        reward = reward or self.rewarder(state, action)

        self.synapses.reward(reward)

        self.callback.network_reward(state, action, reward)
        return reward


class ContinuousRLNetwork(RLNetwork):
    """
    Matrix based spiking neural network with functions build
    for RL w/ continuous rewarding. (rwd at every step).
    - Ensure TD loop does not

    Parameters
    ----------
    callback:
        ...
    game: RL, default=None
        Game to pull parameters from.

    Parameter Priorities
    --------------------
    Highest. Parameters passed directly
    -------. Network.config
    Lowest . Game parameters.

    Templating
    ----------
    config: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Usage
    -----
    Create an SNN object, run .reset() then .tick(state) repeatedly until end game.
    """

    def reward(self, state: object, action: object, reward: float = None) -> float:
        """
        Calculate reward and NOT apply to synapse.
        """
        reward = reward or self.rewarder(state, action)

        return reward

    def continuous_reward(self, state: object, reward: float = None) -> float:
        """
        Calculate reward and apply to synapse.
        """
        action = None

        if self.internal_time < self._processing_time:
            reward = 0
            (
                self.rewarder.prev_td,
                self.rewarder.prev_value,
                self.rewarder.prev_reward,
            ) = (0, 0, 0)
            self.callback.network_reward(state, action, reward)
            return

        action = None
        critic_spikes = self.spike_log[-self._processing_time :, -self._n_neurons :]
        reward = reward or self.rewarder(state, critic_spikes)

        self.synapses.reward(reward)

        self.callback.network_reward(state, action, reward)
        return reward

    def tick(self, state: object) -> object:
        """
        Act based on input data.

        Parameters
        ----------
        state: immutable
            The current game input.

        Returns
        -------
        Corresponding output.
        """
        polarities = np.append(np.ones(self._n_inputs), self.neurons.polarities)

        self._spike_log[: self.synapses._stdp_window] = self._spike_log[
            -self.synapses._stdp_window :
        ]
        normalized_spike_log = self._spike_log.astype(np.bool_)

        self.inputs.update(state)

        if self.modifiers is not None:
            for modifier in self.modifiers:
                modifier.update(self)

        for i in range(self._processing_time):
            self.internal_time += 1

            spikes = np.append(self.inputs(), self.neurons >= self._firing_threshold)

            self._spike_log[self.synapses._stdp_window + i] = spikes
            normalized_spike_log[self.synapses._stdp_window + i] = spikes.astype(
                np.bool_
            )

            self.neurons.update()
            self.synapses.update(
                normalized_spike_log[i : i + self.synapses._stdp_window], polarities
            )

            self.neurons += np.sum(
                self.synapses.weights * spikes.reshape((-1, 1)), axis=0
            )

            self.continuous_reward(state, None)

        outputs = self._spike_log[-self._processing_time :, -self._n_outputs :][::-1]
        output = self.readout(outputs)

        self.callback.network_tick(state, output)
        return output


class FlorianSNN(RLNetwork):
    """
    Matrix based spiking neural network w/ florian2007 reward scheme.
    """

    NECESSARY_KEYS = deepcopy(RLNetwork.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "florian_reward": "float Reward value for florian reward scheme.",
            "florian_punish": "float Punish value for florian punish scheme.",
        }
    )

    def tick(self, state) -> int:
        """
        Act based on state.

        Parameters
        ----------
        state: immutable
            The current game state.

        Returns
        -------
        Action chosen.
        """
        polarities = np.append(np.ones(self._n_inputs), self.neurons.polarities)

        self._spike_log[: self.synapses._stdp_window] = self._spike_log[
            -self.synapses._stdp_window :
        ]
        normalized_spike_log = self._spike_log.astype(np.bool_)

        self.inputs.update(state)

        if self.modifiers is not None:
            for modifier in self.modifiers:
                modifier.update(self)

        for i in range(self._processing_time):
            self.internal_time += 1

            spikes = np.append(self.inputs(), self.neurons >= self._firing_threshold)

            self._spike_log[self.synapses._stdp_window + i] = spikes
            normalized_spike_log[self.synapses._stdp_window + i] = spikes.astype(
                np.bool_
            )

            self.neurons.update()
            self.synapses.update(
                normalized_spike_log[i : i + self.synapses._stdp_window], polarities
            )

            self.neurons += np.sum(
                self.synapses.weights * spikes.reshape((-1, 1)), axis=0
            )

            # Rewards recieved during timestep following the output spike
            if spikes.size and spikes[-1]:
                expected = np.sum(state) % 2

                self.reward(
                    state,
                    None,
                    reward=self._florian_reward if expected else self._florian_punish,
                )

        outputs = self._spike_log[-self._processing_time :, -self._n_outputs :][::-1]
        output = self.readout(outputs)

        self.callback.network_tick(state, output)
        return output
