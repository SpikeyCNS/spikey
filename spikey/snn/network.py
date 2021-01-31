"""
The foundation for building and handling spiking neural networks.
Network serves as the container and manager of all SNN parts like
the neurons, synapses, reward function, ... It is designed to
interact with an RL environment.

There are multiple Network implementations, one for generic usage
and two for different types of reinforcement learning tasks.
"""
from copy import deepcopy
import numpy as np


class Network:
    """
    The foundation for building and handling spiking neural networks.
    Network serves as the container and manager of all SNN parts like
    the neurons, synapses, reward function, ... It is designed to
    interact with an RL environment.

    Note: There are a few types of Networks for different uses, this
    one is the base template for any generic usage.

    Parameters
    ----------
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to for logging.
    game: RL, default=None
        The environment the network will be interacting with, parameter
        is to allow network to pull relevant parameters in init.
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Parameter Priorities
    --------------------
    Network parameters to fill NECESSARY_KEYS may come from a variety of
    sources, the overloading priority is as follows.

    Highest: Passed directly into constructor(kwargs).
    Middle : Network.config defined before init is called.
    Lowest : Game parameters being shared by passing the game to init.

    Templating
    ----------
    If Network is templated, default parameter values can be set via
    member variables config and parts that are interpreted similarly
    to kwargs but with a lower priority.

    config: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Usage
    -----
    ```python
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
        "firing_threshold": 16,
        # + all part parameters, see Network.list_necessary_keys(**parts)
    }
    config = {**parts, **params}

    game = Logic(preset="XOR", **config)
    network = Network(game=game, **config)

    for _ in range(experiment_params["n_episodes"]):
        network.reset()
        state = game.reset()

        for s in range(experiment_params["len_episode"]):
            action = network.tick(state)

            state, _, done, __ = game.step(action)

            if done:
                break
    ```

    ```python
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
        config = {
            "n_inputs": 10,
            "n_outputs": 10,
            "n_neurons": 50,
            "processing_time": 200,
            "firing_threshold": 16,
            # + all part parameters, see Network.list_necessary_keys(**parts)
        }

    kwargs = {
        "n_neurons": 100,  # Overrides n_neurons in network_template.config
    }

    game = Logic(preset="XOR", **kwargs)
    network = network_template(game=game, **kwargs)

    for _ in range(experiment_params["n_episodes"]):
        network.reset()
        state = game.reset()

        for s in range(experiment_params["len_episode"]):
            action = network.tick(state)

            state, _, done, __ = game.step(action)

            if done:
                break
    ```
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
        Neuron spike log over processing_time with spike_log[-1] being most recent.
        """
        try:
            return self._spike_log[-self._processing_time :]
        except TypeError:
            return None

    @classmethod
    def list_necessary_keys(cls, **kwargs):
        """
        Print list of all required keys for this network configuration with all parts.

        Parameters
        ----------
        kwargs: dict
            Overrides to parts dictionary.
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

        Usage
        -----
        ```python
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
            config = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                "firing_threshold": 16,
                # + all part parameters, see Network.list_necessary_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.config
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                if done:
                    break
        ```
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
        Determine network response to given stimulus.

        Parameters
        ----------
        state: any
            Current environment state.

        Returns
        -------
        any Network response to stimulus.

        Usage
        -----
        ```python
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
            config = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                "firing_threshold": 16,
                # + all part parameters, see Network.list_necessary_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.config
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                if done:
                    break
        ```
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
    The foundation for building and handling spiking neural networks.
    Network serves as the container and manager of all SNN parts like
    the neurons, synapses, reward function, ... It is designed to
    interact with an RL environment.

    Note: There are a few types of Networks for different uses, this
    one is the base for reinforcement learning with SNNs giving one
    reward per game update(see ContinuousRLNetwork reward for per network
    step).

    Parameters
    ----------
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to for logging.
    game: RL, default=None
        The environment the network will be interacting with, parameter
        is to allow network to pull relevant parameters in init.
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Parameter Priorities
    --------------------
    Network parameters to fill NECESSARY_KEYS may come from a variety of
    sources, the overloading priority is as follows.

    Highest: Passed directly into constructor(kwargs).
    Middle : Network.config defined before init is called.
    Lowest : Game parameters being shared by passing the game to init.

    Templating
    ----------
    If Network is templated, default parameter values can be set via
    member variables config and parts that are interpreted similarly
    to kwargs but with a lower priority.

    config: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Usage
    -----
    ```python
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
        "firing_threshold": 16,
        # + all part parameters, see Network.list_necessary_keys(**parts)
    }
    config = {**parts, **params}

    game = Logic(preset="XOR", **config)
    network = RLNetwork(game=game, **config)

    for _ in range(experiment_params["n_episodes"]):
        network.reset()
        state = game.reset()

        for s in range(experiment_params["len_episode"]):
            action = network.tick(state)

            state, _, done, __ = game.step(action)

            reward = network.reward(state, action)

            if done:
                break
    ```

    ```python
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
        config = {
            "n_inputs": 10,
            "n_outputs": 10,
            "n_neurons": 50,
            "processing_time": 200,
            "firing_threshold": 16,
            # + all part parameters, see Network.list_necessary_keys(**parts)
        }

    kwargs = {
        "n_neurons": 100,  # Overrides n_neurons in network_template.config
    }

    game = Logic(preset="XOR", **kwargs)
    network = network_template(game=game, **kwargs)

    for _ in range(experiment_params["n_episodes"]):
        network.reset()
        state = game.reset()

        for s in range(experiment_params["len_episode"]):
            action = network.tick(state)

            state, _, done, __ = game.step(action)

            reward = network.reward(state, action)

            if done:
                break
    ```
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

        Usage
        -----
        ```python
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
            config = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                "firing_threshold": 16,
                # + all part parameters, see Network.list_necessary_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.config
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                reward = network.reward(state, action)

                if done:
                    break
        ```
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

    Note: There are a few types of Networks for different uses, this
    one is the base for reinforcement learning with SNNs giving reward
    at every network step(see RLNetwork for reward per game step).

    Parameters
    ----------
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to for logging.
    game: RL, default=None
        The environment the network will be interacting with, parameter
        is to allow network to pull relevant parameters in init.
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Parameter Priorities
    --------------------
    Network parameters to fill NECESSARY_KEYS may come from a variety of
    sources, the overloading priority is as follows.

    Highest: Passed directly into constructor(kwargs).
    Middle : Network.config defined before init is called.
    Lowest : Game parameters being shared by passing the game to init.

    Templating
    ----------
    If Network is templated, default parameter values can be set via
    member variables config and parts that are interpreted similarly
    to kwargs but with a lower priority.

    config: dict
        Key-value pairs for everything in NECESSARY_KEYS for all objects.
    parts: dict
        Parts that make up network, see NECESSARY_PARTS.

    Usage
    -----
    ```python
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
        "firing_threshold": 16,
        # + all part parameters, see Network.list_necessary_keys(**parts)
    }
    config = {**parts, **params}

    game = Logic(preset="XOR", **config)
    network = RLNetwork(game=game, **config)

    for _ in range(experiment_params["n_episodes"]):
        network.reset()
        state = game.reset()

        for s in range(experiment_params["len_episode"]):
            action = network.tick(state)

            state, _, done, __ = game.step(action)

            # Calculated reward per env step, does not affect network
            # Actual rewarding handled in ContinuousRLNetwork.tick().
            reward = network.reward(state, action)

            if done:
                break
    ```

    ```python
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
        config = {
            "n_inputs": 10,
            "n_outputs": 10,
            "n_neurons": 50,
            "processing_time": 200,
            "firing_threshold": 16,
            # + all part parameters, see Network.list_necessary_keys(**parts)
        }

    kwargs = {
        "n_neurons": 100,  # Overrides n_neurons in network_template.config
    }

    game = Logic(preset="XOR", **kwargs)
    network = network_template(game=game, **kwargs)

    for _ in range(experiment_params["n_episodes"]):
        network.reset()
        state = game.reset()

        for s in range(experiment_params["len_episode"]):
            action = network.tick(state)

            state, _, done, __ = game.step(action)

            # Calculated reward per env step, does not affect network
            # Actual rewarding handled in ContinuousRLNetwork.tick().
            reward = network.reward(state, action)

            if done:
                break
    ```
    """

    def reward(self, state: object, action: object, reward: float = None) -> float:
        """
        Calculate reward per environment step and DON'T apply it to anywhere.

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

        Usage
        -----
        ```python
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
            config = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                "firing_threshold": 16,
                # + all part parameters, see Network.list_necessary_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.config
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                # Calculated reward per env step, does not affect network
                # Actual rewarding handled in ContinuousRLNetwork.tick().
                reward = network.reward(state, action)

                if done:
                    break
        ```
        """
        reward = reward or self.rewarder(state, action)

        return reward

    def continuous_reward(self, state: object, reward: float = None) -> float:
        """
        Calculate reward and apply to synapses.

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

        Usage
        -----
        ```python
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
            config = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                "firing_threshold": 16,
                # + all part parameters, see Network.list_necessary_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.config
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                # Calculated reward per env step, does not affect network
                # Actual rewarding handled in ContinuousRLNetwork.tick().
                reward = network.reward(state, action)

                if done:
                    break
        ```
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
        Determine network response to given stimulus.

        Parameters
        ----------
        state: any
            Current environment state.

        Returns
        -------
        any Network response to stimulus.

        Usage
        -----
        ```python
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
            config = {
                "n_inputs": 10,
                "n_outputs": 10,
                "n_neurons": 50,
                "processing_time": 200,
                "firing_threshold": 16,
                # + all part parameters, see Network.list_necessary_keys(**parts)
            }

        kwargs = {
            "n_neurons": 100,  # Overrides n_neurons in network_template.config
        }

        game = Logic(preset="XOR", **kwargs)
        network = network_template(game=game, **kwargs)

        for _ in range(experiment_params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(experiment_params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                # Calculated reward per env step, does not affect network
                # Actual rewarding handled in ContinuousRLNetwork.tick().
                reward = network.reward(state, action)

                if done:
                    break
        ```
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
    (Deprecated)
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
