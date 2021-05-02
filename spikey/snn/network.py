"""
The foundation for building and handling spiking neural networks.
Network serves as the container and manager of all SNN parts like
the neurons, synapses, reward function, ... It is designed to
interact with an RL environment.

There are multiple Network implementations, one for generic usage
and two for different types of reinforcement learning tasks.
"""
from spikey.module import Module, Key
from copy import deepcopy
import numpy as np
from spikey.core import ExperimentCallback


class Network(Module):
    """
    The foundation for building and handling spiking neural networks.
    Network serves as the container and manager of all SNN parts like
    the neurons, synapses, reward function, ... It is designed to
    interact with an RL environment.

    .. note::
        There are a few types of Networks for different uses, this
        one is the base template for any generic usage.

    Parameter Priorities

    Network parameters to fill NECESSARY_KEYS may come from a variety of
    sources, the overloading priority is as follows.

    Highest: Passed directly into constructor(kwargs).
    Middle : Network.keys defined before init is called.
    Lowest : Game parameters being shared by passing the game to init.

    Templating

    If Network is templated, default parameter values can be set via
    member variables keys and parts that are interpreted similarly
    to kwargs but with a lower priority.

    keys: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.


    Parameters
    ----------
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to for logging.
    game: RL, default=None
        The environment the network will be interacting with, parameter
        is to allow network to pull relevant parameters in init.
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        experiment_params = {
            "n_episodes": 100,
            "len_episode": 200,
        }

        parts = {
            "inputs": snn.input.Input,
            "neurons": snn.neuron.Neuron,
            "weights": snn.weight.Weight,
            "synapses": snn.synapse.Synapse,
            "readout": snn.readout.Readout,
            "modifiers": None, # [snn.modifier.Modifier,]
        }
        params = {
            "n_inputs": 10,
            "n_outputs": 10,
            "n_neurons": 50,
            "processing_time": 200,
            # + all part parameters, see Network.list_keys(**parts)
        }
        config = {**parts, **params}

        game = Logic(preset="XOR", **config)
        network = Network(game=game, **config)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)
                state_next, _, done, __ = game.step(action)
                reward = network.reward(state, action)
                state = state_next

                if done:
                    break

    .. code-block:: python

        experiment_params = {
            "n_episodes": 100,
            "len_episode": 200,
        }

        class network_template(Network):
            parts = {
                "inputs": snn.input.Input,
                "neurons": snn.neuron.Neuron,
                "weights": snn.weight.Weight,
                "synapses": snn.synapse.Synapse,
                "readout": snn.readout.Readout,
                "modifiers": None, # [snn.modifier.Modifier,]
            }
            keys = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                # + all part parameters, see Network.list_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.keys
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)
                state_next, _, done, __ = game.step(action)
                reward = network.reward(state, action)
                state = state_next

                if done:
                    break
    """

    NECESSARY_KEYS = [
        Key("n_inputs", "Number input neurons.", int),
        Key(
            "n_outputs", "n_outputs = n_neurons - n_body Number of output neurons.", int
        ),
        Key("n_neurons", "Number of neurons in the network.", int),
        Key(
            "processing_time",
            "Number of network timesteps per game timestep.",
            int,
        ),
    ]
    NECESSARY_PARTS = [
        Key("inputs", "snn.input.Input"),
        Key("neurons", "snn.neuron.Neuron"),
        Key("weights", "snn.weight.Weight"),
        Key("synapses", "snn.synapse.Synapse"),
        Key("readout", "snn.readout.Readout"),
        Key("modifiers", "list of snn.modifier.Modifier", default=None),
    ]

    def __init__(
        self,
        callback: object = None,
        game: object = None,
        **kwargs,
    ):
        if not hasattr(self, "parts"):
            self.parts = {}
        else:
            self.parts = deepcopy(type(self).parts)
        if "modifiers" not in self.parts:
            self.parts["modifiers"] = None
        for key in self.NECESSARY_PARTS:
            if key in kwargs:
                self.parts[key] = kwargs[key]
        self._params = {} if game is None else deepcopy(game.params)
        if hasattr(self, "keys"):
            self._params.update(self.keys)
        self._params.update(kwargs)

        super().__init__(**self._params)

        self.callback = callback or ExperimentCallback()

        self._init_parts()

        self.internal_time = self._spike_log = None

        self.callback.network_init(self)

    def _init_parts(self):
        for key in self.NECESSARY_PARTS:
            name = key.name if isinstance(key, Key) else key

            if name in self.parts:
                part = self.parts[name]
            elif isinstance(key, Key) and hasattr(key, "default"):
                part = key.default
            else:
                raise ValueError(f"No value given for key {name}!")

            if name == "synapses":
                value = part(self.weights, **self.params)
            elif part is None:
                value = part
            else:
                value = part(**self.params)

            setattr(self, name, value)

        self.synapses.weights = self.weights

    def train(self):
        """
        Set the module to training mode, enabled by default.
        """
        self.training = True
        for key in self.NECESSARY_PARTS:
            name = key.name if isinstance(key, Key) else key
            try:
                getattr(self, name).train()
            except AttributeError:
                pass

    def eval(self):
        """
        Set the module to evaluation mode, disabled by default.
        """
        self.training = False
        for key in self.NECESSARY_PARTS:
            name = key.name if isinstance(key, Key) else key
            try:
                getattr(self, name).eval()
            except AttributeError:
                pass

    @property
    def params(self) -> dict:
        """
        Read only configuration of network.
        """
        return deepcopy(self._params)

    @property
    def spike_log(self) -> np.bool:
        """
        Neuron spike log over processing_time with spike_log[-1] being most recent.
        """
        try:
            return self._spike_log[-self._processing_time :]
        except TypeError:
            return None

    @classmethod
    def list_keys(cls, **parts):
        """
        Print list of all required keys for the Network and
        its parts.
        """
        if isinstance(cls.NECESSARY_KEYS, dict):
            KEYS = {}
        else:
            KEYS = deepcopy(cls.NECESSARY_KEYS)
        for part in parts.values():
            if not hasattr(part, "NECESSARY_KEYS"):
                continue
            if isinstance(KEYS, dict):
                KEYS.update(part.NECESSARY_KEYS)
            else:
                KEYS.extend([p for p in part.NECESSARY_KEYS if p not in KEYS])
        if isinstance(cls.NECESSARY_KEYS, dict):
            KEYS.update(cls.NECESSARY_KEYS)

        print("{")
        for key in KEYS:
            if isinstance(key, Key):
                print(f"\t{str(key)},")
            else:
                desc = cls.NECESSARY_KEYS[key]
                print(f"\t{key}: {desc},")

        print("}")

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        network = cls.__new__(cls)
        memo[id(self)] = network
        for k, v in self.__dict__.items():
            setattr(network, k, deepcopy(v, memo))
        network._init_parts()
        return network

    def reset(self):
        """
        Set network to initial state.

        Examples
        --------

        .. code-block:: python

            experiment_params = {
                "n_episodes": 100,
                "len_episode": 200,
            }

            class network_template(Network):
                parts = {
                    "inputs": snn.input.Input,
                    "neurons": snn.neuron.Neuron,
                    "weights": snn.weight.Weight,
                    "synapses": snn.synapse.Synapse,
                    "readout": snn.readout.Readout,
                    "modifiers": None, # [snn.modifier.Modifier,]
                }
                keys = {
                    "n_inputs": 10,
                    "n_outputs": 10,
                    "n_neurons": 50,
                    "processing_time": 200,
                    # + all part parameters, see Network.list_keys(**parts)
                }

            kwargs = {
                "n_neurons": 100,  # Overrides n_neurons in network_template.keys
            }

            game = Logic(preset="XOR", **kwargs)
            network = network_template(game=game, **kwargs)

            for _ in range(experiment_params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(experiment_params["len_episode"]):
                    action = network.tick(state)
                    state_next, _, done, __ = game.step(action)
                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break
        """
        self.internal_time = 0

        self.neurons.reset()
        self.synapses.reset()
        if hasattr(self, "rewarder"):
            self.rewarder.reset()
        self.readout.reset()
        self.inputs.reset()
        for modifier in self.modifiers or []:
            modifier.reset()

        self._spike_log = np.zeros(
            (
                self.synapses._stdp_window + self._processing_time,
                self._n_inputs + self._n_neurons,
            ),
            dtype=np.float16,
        )

        self.callback.network_reset()

    def _process_step(self, i: int, state: object):
        """
        Execute one processing step.

        Parameters
        ----------
        i: int
            Current processing timestep.
        state: any
            Current environment state.
        """
        self.internal_time += 1

        spikes = np.append(self.inputs(), self.neurons())

        self._spike_log[self.synapses._stdp_window + i] = spikes
        self._normalized_spike_log[self.synapses._stdp_window + i] = spikes.astype(
            np.bool_
        )

        self.neurons += np.sum(self.synapses.weights * spikes.reshape((-1, 1)), axis=0)

        self.synapses.update(
            self._normalized_spike_log[i : i + self.synapses._stdp_window],
            self._polarities,
        )

    def tick(self, state: object) -> object:
        """
        Determine network response to given stimulus.

        Parameters
        ----------
        state: any
            Current environment state.

        Returns
        -------
        any Network response to stimulus.

        Examples
        --------

        .. code-block:: python

            experiment_params = {
                "n_episodes": 100,
                "len_episode": 200,
            }

            class network_template(Network):
                parts = {
                    "inputs": snn.input.Input,
                    "neurons": snn.neuron.Neuron,
                    "weights": snn.weight.Weight,
                    "synapses": snn.synapse.Synapse,
                    "readout": snn.readout.Readout,
                    "modifiers": None, # [snn.modifier.Modifier,]
                }
                keys = {
                    "n_inputs": 10,
                    "n_outputs": 10,
                    "n_neurons": 50,
                    "processing_time": 200,
                    # + all part parameters, see Network.list_keys(**parts)
                }

            kwargs = {
                "n_neurons": 100,  # Overrides n_neurons in network_template.keys
            }

            game = Logic(preset="XOR", **kwargs)
            network = network_template(game=game, **kwargs)

            for _ in range(experiment_params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(experiment_params["len_episode"]):
                    action = network.tick(state)
                    state_next, _, done, __ = game.step(action)
                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break
        """
        self._polarities = np.append(self.inputs.polarities, self.neurons.polarities)

        self._spike_log[: self.synapses._stdp_window] = self._spike_log[
            -self.synapses._stdp_window :
        ]
        self._normalized_spike_log = self._spike_log.astype(np.bool_)

        self.inputs.update(state)

        if self.modifiers is not None:
            for modifier in self.modifiers:
                modifier.update(self)

        for i in range(self._processing_time):
            self._process_step(i, state)

        outputs = self._spike_log[-self._processing_time :, -self._n_outputs :]
        output = self.readout(outputs)

        self.callback.network_tick(state, output)
        return output


class RLNetwork(Network):
    """
    The foundation for building and handling spiking neural networks.
    Network serves as the container and manager of all SNN parts like
    the neurons, synapses, reward function, ... It is designed to
    interact with an RL environment.

    .. note::
        There are a few types of Networks for different uses, this
        one is the base for reinforcement learning with SNNs giving one
        reward per game update(see ContinuousRLNetwork reward for per network
        step).

    Parameter Priorities

    Network parameters to fill NECESSARY_KEYS may come from a variety of
    sources, the overloading priority is as follows.

    Highest: Passed directly into constructor(kwargs).
    Middle : Network.keys defined before init is called.
    Lowest : Game parameters being shared by passing the game to init.

    Templating

    If Network is templated, default parameter values can be set via
    member variables keys and parts that are interpreted similarly
    to kwargs but with a lower priority.

    keys: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Parameters
    ----------
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to for logging.
    game: RL, default=None
        The environment the network will be interacting with, parameter
        is to allow network to pull relevant parameters in init.
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        experiment_params = {
            "n_episodes": 100,
            "len_episode": 200,
        }

        parts = {
            "inputs": snn.input.Input,
            "neurons": snn.neuron.Neuron,
            "weights": snn.weight.Weight,
            "synapses": snn.synapse.Synapse,
            "readout": snn.readout.Readout,
            "rewarder": snn.reward.Reward,
            "modifiers": None, # [snn.modifier.Modifier,]
        }
        params = {
            "n_inputs": 10,
            "n_outputs": 10,
            "n_neurons": 50,
            "processing_time": 200,
            # + all part parameters, see Network.list_keys(**parts)
        }
        config = {**parts, **params}

        game = Logic(preset="XOR", **config)
        network = RLNetwork(game=game, **config)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)
                state_next, _, done, __ = game.step(action)
                reward = network.reward(state, action)
                state = state_next

                if done:
                    break
    .. code-block:: python

        experiment_params = {
            "n_episodes": 100,
            "len_episode": 200,
        }

        class network_template(RLNetwork):
            parts = {
                "inputs": snn.input.Input,
                "neurons": snn.neuron.Neuron,
                "weights": snn.weight.Weight,
                "synapses": snn.synapse.Synapse,
                "readout": snn.readout.Readout,
                "rewarder": snn.reward.Reward,
                "modifiers": None, # [snn.modifier.Modifier,]
            }
            keys = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                # + all part parameters, see Network.list_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.keys
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)
                state_next, _, done, __ = game.step(action)
                reward = network.reward(state, action)
                state = state_next

                if done:
                    break
    """

    NECESSARY_PARTS = Network.extend_keys(
        [
            Key("rewarder", "snn.reward.Reward"),
        ],
        base="NECESSARY_PARTS",
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
        If reward given as parameter, apply reward to synapses.
        Otherwise rewarder calculates based on state and action, then applies to synapses.
        Called once per game step.

        Parameters
        ----------
        state: any
            State of environment where action was taken.
        action: any
            Action taken by network in response to state.
        reward: float, default=None
            Reward to give network, if None it will be determined by the rewarder.

        Returns
        -------
        float Reward given to network.

        Examples
        --------

        .. code-block:: python

            experiment_params = {
                "n_episodes": 100,
                "len_episode": 200,
            }

            class network_template(RLNetwork):
                parts = {
                    "inputs": snn.input.Input,
                    "neurons": snn.neuron.Neuron,
                    "weights": snn.weight.Weight,
                    "synapses": snn.synapse.Synapse,
                    "readout": snn.readout.Readout,
                    "modifiers": None, # [snn.modifier.Modifier,]
                }
                keys = {
                    "n_inputs": 10,
                    "n_outputs": 10,
                    "n_neurons": 50,
                    "processing_time": 200,
                    # + all part parameters, see Network.list_keys(**parts)
                }

            kwargs = {
                "n_neurons": 100,  # Overrides n_neurons in network_template.keys
            }

            game = Logic(preset="XOR", **kwargs)
            network = network_template(game=game, **kwargs)

            for _ in range(experiment_params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(experiment_params["len_episode"]):
                    action = network.tick(state)
                    state_next, _, done, __ = game.step(action)
                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break
        """
        reward = reward or self.rewarder(state, action)

        self.synapses.reward(reward)

        self.callback.network_reward(state, action, reward)
        return reward


class ContinuousRLNetwork(RLNetwork):
    """
    The foundation for building and handling spiking neural networks.
    Network serves as the container and manager of all SNN parts like
    the neurons, synapses, reward function, ... It is designed to
    interact with an RL environment.

    .. note::
        There are a few types of Networks for different uses, this
        one is the base for reinforcement learning with SNNs giving reward
        at every network step(see RLNetwork for reward per game step).

    Parameter Priorities

    Network parameters to fill NECESSARY_KEYS may come from a variety of
    sources, the overloading priority is as follows.

    Highest: Passed directly into constructor(kwargs).
    Middle : Network.keys defined before init is called.
    Lowest : Game parameters being shared by passing the game to init.

    Templating

    If Network is templated, default parameter values can be set via
    member variables keys and parts that are interpreted similarly
    to kwargs but with a lower priority.

    keys: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Parameters
    ----------
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to for logging.
    game: RL, default=None
        The environment the network will be interacting with, parameter
        is to allow network to pull relevant parameters in init.
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        experiment_params = {
            "n_episodes": 100,
            "len_episode": 200,
        }

        parts = {
            "inputs": snn.input.Input,
            "neurons": snn.neuron.Neuron,
            "weights": snn.weight.Weight,
            "synapses": snn.synapse.Synapse,
            "readout": snn.readout.Readout,
            "rewarder": snn.reward.Reward,
            "modifiers": None, # [snn.modifier.Modifier,]
        }
        params = {
            "n_inputs": 10,
            "n_outputs": 10,
            "n_neurons": 50,
            "processing_time": 200,
            # + all part parameters, see Network.list_keys(**parts)
        }
        config = {**parts, **params}

        game = Logic(preset="XOR", **config)
        network = RLNetwork(game=game, **config)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state_next, _, done, __ = game.step(action)

                # Calculated reward per env step, does not affect network
                # Actual rewarding handled in ContinuousRLNetwork.tick().
                reward = network.reward(state, action)
                state = state_next

                if done:
                    break

    .. code-block:: python

        experiment_params = {
            "n_episodes": 100,
            "len_episode": 200,
        }

        class network_template(ContinuousRLNetwork):
            parts = {
                "inputs": snn.input.Input,
                "neurons": snn.neuron.Neuron,
                "weights": snn.weight.Weight,
                "synapses": snn.synapse.Synapse,
                "readout": snn.readout.Readout,
                "rewarder": snn.reward.Reward,
                "modifiers": None, # [snn.modifier.Modifier,]
            }
            keys = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                # + all part parameters, see Network.list_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.keys
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state_next, _, done, __ = game.step(action)

                # Calculated reward per env step, does not affect network
                # Actual rewarding handled in ContinuousRLNetwork.tick().
                reward = network.reward(state, action)
                state = state_next

                if done:
                    break
    """

    NECESSARY_KEYS = RLNetwork.extend_keys(
        [
            Key(
                "continuous_rwd_action",
                "f(network, state)->any Function to get action parameter for rewarder when using continuous_reward.",
            )
        ]
    )

    def reward(self, state: object, action: object, reward: float = None) -> float:
        """
        If reward given as parameter and DON'T apply reward to synapses.
        Otherwise rewarder calculates based on state and action and DON'T then applies to synapses.
        Called once per game step.

        Parameters
        ----------
        state: any
            State of environment where action was taken.
        action: any
            Action taken by network in response to state.
        reward: float, default=None
            Reward already calculated, if None it will be determined by the rewarder.

        Returns
        -------
        float Reward calculated for taking action in state.

        Examples
        --------

        .. code-block:: python

            experiment_params = {
                "n_episodes": 100,
                "len_episode": 200,
            }

            class network_template(ContinuousRLNetwork):
                parts = {
                    "inputs": snn.input.Input,
                    "neurons": snn.neuron.Neuron,
                    "weights": snn.weight.Weight,
                    "synapses": snn.synapse.Synapse,
                    "readout": snn.readout.Readout,
                    "modifiers": None, # [snn.modifier.Modifier,]
                }
                keys = {
                    "n_inputs": 10,
                    "n_outputs": 10,
                    "n_neurons": 50,
                    "processing_time": 200,
                    # + all part parameters, see Network.list_keys(**parts)
                }

            kwargs = {
                "n_neurons": 100,  # Overrides n_neurons in network_template.keys
            }

            game = Logic(preset="XOR", **kwargs)
            network = network_template(game=game, **kwargs)

            for _ in range(experiment_params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(experiment_params["len_episode"]):
                    action = network.tick(state)

                    state_next, _, done, __ = game.step(action)

                    # Calculated reward per env step, does not affect network
                    # Actual rewarding handled in ContinuousRLNetwork.tick().
                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break
        """
        self.callback.network_reward(state, action, reward)
        return reward

    def continuous_reward(self, state: object, reward: float = None) -> float:
        """
        If reward given as parameter, apply reward to synapses.
        Otherwise rewarder calculates based on state and action, then applies to synapses.
        Continuous reward meant to be applied per network step.

        Parameters
        ----------
        state: any
            State of environment where action was taken.
        action: any
            Action taken by network in response to state.
        reward: float, default=None
            Reward to give network, if None it will be determined by the rewarder.

        Returns
        -------
        float Reward given to network.

        Examples
        --------

        .. code-block:: python

            experiment_params = {
                "n_episodes": 100,
                "len_episode": 200,
            }

            class network_template(ContinuousRLNetwork):
                parts = {
                    "inputs": snn.input.Input,
                    "neurons": snn.neuron.Neuron,
                    "weights": snn.weight.Weight,
                    "synapses": snn.synapse.Synapse,
                    "readout": snn.readout.Readout,
                    "modifiers": None, # [snn.modifier.Modifier,]
                }
                keys = {
                    "n_inputs": 10,
                    "n_outputs": 10,
                    "n_neurons": 50,
                    "processing_time": 200,
                    # + all part parameters, see Network.list_keys(**parts)
                }

            kwargs = {
                "n_neurons": 100,  # Overrides n_neurons in network_template.keys
            }

            game = Logic(preset="XOR", **kwargs)
            network = network_template(game=game, **kwargs)

            for _ in range(experiment_params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(experiment_params["len_episode"]):
                    action = network.tick(state)

                    state_next, _, done, __ = game.step(action)

                    # Calculated reward per env step, does not affect network
                    # Actual rewarding handled in ContinuousRLNetwork.tick().
                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break
        """
        action = self._continuous_rwd_action(self, state)

        reward = reward or self.rewarder(state, action)

        self.synapses.reward(reward)

        self.callback.network_continuous_reward(state, action, reward)
        return reward

    def tick(self, state: object) -> object:
        """
        Determine network response to given stimulus.

        Parameters
        ----------
        state: any
            Current environment state.

        Returns
        -------
        any Network response to stimulus.

        Examples
        --------

        .. code-block:: python

            experiment_params = {
                "n_episodes": 100,
                "len_episode": 200,
            }

            class network_template(ContinuousRLNetwork):
                parts = {
                    "inputs": snn.input.Input,
                    "neurons": snn.neuron.Neuron,
                    "weights": snn.weight.Weight,
                    "synapses": snn.synapse.Synapse,
                    "readout": snn.readout.Readout,
                    "modifiers": None, # [snn.modifier.Modifier,]
                }
                keys = {
                    "n_inputs": 10,
                    "n_outputs": 10,
                    "n_neurons": 50,
                    "processing_time": 200,
                    # + all part parameters, see Network.list_keys(**parts)
                }

            kwargs = {
                "n_neurons": 100,  # Overrides n_neurons in network_template.keys
            }

            game = Logic(preset="XOR", **kwargs)
            network = network_template(game=game, **kwargs)

            for _ in range(experiment_params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(experiment_params["len_episode"]):
                    action = network.tick(state)

                    state_next, _, done, __ = game.step(action)

                    # Calculated reward per env step, does not affect network
                    # Actual rewarding handled in ContinuousRLNetwork.tick().
                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break
        """

        self._polarities = np.append(self.inputs.polarities, self.neurons.polarities)

        self._spike_log[: self.synapses._stdp_window] = self._spike_log[
            -self.synapses._stdp_window :
        ]
        self._normalized_spike_log = self._spike_log.astype(np.bool_)

        self.inputs.update(state)

        if self.modifiers is not None:
            for modifier in self.modifiers:
                modifier.update(self)

        for i in range(self._processing_time):
            self._process_step(i, state)

            self.continuous_reward(state, None)

        outputs = self._spike_log[-self._processing_time :, -self._n_outputs :]
        output = self.readout(outputs)

        self.callback.network_tick(state, output)
        return output
