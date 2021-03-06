"""
Pre-built, reusable training loops that allow users
to train a network on a game individually, repeatedly
or within the meta analysis tools.
"""
from copy import deepcopy

from spikey.module import Module, Key
from spikey.core.callback import ExperimentCallback


class TrainingLoop(Module):
    """
    Template for pre-built, reusable training loops.

    Parameters
    ----------
    network_template: SNN[class]
        Network type to train.
    game_template: RL[class]
        Game type to train.
    callback: ExperimentCallback[class or object], default=ExperimentCallback
        Callback object template or already initialized callback.
    **params: dict
        Network, game and training parameters.

    Usage
    -----
    ```python
    experiment = TrainingLoop(Network, RL, **config)
    experiment.reset()

    network, game, results, info = experiment()
    ```
    """
    NECESSARY_KEYS = []

    def __init__(self, network_template: type, game_template: type, callback: ExperimentCallback = None, **params: dict):
        self.network_template = network_template
        self.game_template = game_template
        self.callback = callback or ExperimentCallback

        if not len(params):
            print(f"WARNING: No values given as {type(self)} params!")
        self.params = {}
        if hasattr(self.network_template, "config"):
            self.params.update(deepcopy(self.network_template.config))
        self.params.update(params)

        super().__init__(**self.params)

    def reset(
        self,
        network_template: type = None,
        game_template: type = None,
        callback: ExperimentCallback = None,
        params: dict=None,
        **kwparams: dict,
    ):
        """
        Reset, optionally override network_template, game_template and parameters.

        Parameters
        ----------
        network_template: Network, default=None
            New network template.
        game_template: RL, default=None
            New game template.
        callback: ExperimentCallback[class or object], default=None
            Callback object template or already initialized callback.
        **params: dict
            Updates to network, game and training parameters.

        Usage
        -----
        ```python
        experiment = TrainingLoop(Network, RL, **config)
        experiment.reset()
        ```
        """
        if network_template is not None:
            self.network_template = network_template
        if game_template is not None:
            self.game_template = game_template
        if callback is not None:
            self.callback = callback
        if params is not None:
            self.params.update(params)
        if kwparams is not None:
            self.params.update(kwparams)

    def _init_callback(self):
        """
        Initialize callback object for TrainingLoop.
        """
        if type(self.callback) == type:
            callback = self.callback(**self.params)
        else:
            callback = self.callback
        return callback

    def __call__(self) -> (object, object, dict, dict):
        """
        Run training loop a single time.

        Returns
        -------
        network: Network, game: RL, results: dict, info: dict.

        Usage
        -----
        ```python
        experiment = TrainingLoop(Network, RL, **config)
        experiment.reset()

        network, game, results, info = experiment()
        ```
        """
        raise NotImplementedError(f"Call not implemented in {type(self)}.")


class GenericLoop(TrainingLoop):
    """
    Generic reinforcement learning training loop.

    ```
    for ep in n_episodes:
        while not done or until i == len_episode:
            action = network.tick(state)
            state_next, _, done, __ = game.step(action)
            reward = network.reward(state, action)
            state = state_next
    ```

    Parameters
    ----------
    network_template: SNN[class]
        Network type to train.
    game_template: RL[class]
        Game type to train.
    params: dict
        Network, game and training parameters.

    Usage
    -----
    ```python
    experiment = GenericLoop(Network, RL, **config)
    experiment.reset()

    network, game, results, info = experiment()
    ```
    """

    NECESSARY_KEYS = TrainingLoop.extend_keys(
        [
            Key("n_episodes", "Number of episodes to run in the experiment.", int),
            Key("len_episode", "Number of environment timesteps in each episode", int),
        ]
    )

    def __call__(self) -> (object, object, dict, dict):
        """
        Run training loop a single time.

        Returns
        -------
        network: Network, game: RL, results: dict, info: dict.

        Usage
        -----
        ```python
        experiment = TrainingLoop(Network, RL, **config)
        experiment.reset()

        network, game, results, info = experiment()
        ```
        """
        callback = self._init_callback()
        callback.reset({key.name if isinstance(key, Key) else key: self.params[key] for key in self.NECESSARY_KEYS})
        game = self.game_template(callback=callback, **self.params)
        network = self.network_template(callback=callback, game=game, **self.params)

        for e in range(self.params["n_episodes"]):
            network.reset()
            state = game.reset()
            state_next = None

            for s in range(self.params["len_episode"]):
                action = network.tick(state)
                state_next, _, done, __ = game.step(action)
                if hasattr(network, "reward") and callable(getattr(network, "reward")):
                    reward = network.reward(state, action)
                state = state_next

                if done:
                    break

        callback.training_end()

        return [*callback]
