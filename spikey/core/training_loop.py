"""
Pre-built, reusable training loops.
"""
from spikey.module import Module
from copy import deepcopy

from spikey.core.callback import RLCallback


class TrainingLoop(Module):
    """
    Template for pre-built, reusable training loops.

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
    experiment = TrainingLoop(Network, RL, **config)
    experiment.reset()

    network, game, results, info = experiment()
    ```
    """

    NECESSARY_KEYS = {}

    def __init__(self, network_template: type, game_template: type, params: dict):
        self.network_template = network_template
        self.game_template = game_template

        self.params = {}
        if hasattr(self.network_template, 'config'):
            self.params.update(deepcopy(self.network_template.config))
        self.params.update(params)

        super().__init__(**self.params)

    def reset(
        self,
        network_template: type = None,
        game_template: type = None,
        params: dict = None,
    ):
        """
        Reset, optionally override network_template, game_template and parameters.

        Parameters
        ----------
        network_template: Network, default=None
            New network template.
        game_template: RL, default=None
            New game template.
        params: dict, default=None
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
        if params is not None:
            self.params.update(params)

    def __call__(self, **kwargs) -> (object, object, dict, dict):
        """
        Run training loop a single time.

        Parameters
        ----------
        kwargs: dict
            Any optional arguments.

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
            state, _, done, __ = game.step(action)
            reward = network.reward(state, action)
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
        {
            "n_episodes": "int Number of episodes to run,",
            "len_episode": "int Length of episode.",
        }
    )

    def __call__(self, **kwargs) -> (object, object, dict, dict):
        """
        Run training loop a single time.

        Parameters
        ----------
        kwargs: dict
            Any optional arguments.

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
        if "callback" in kwargs:
            experiment = kwargs["callback"]
        else:
            experiment = RLCallback(
                **self.params,
                reduced=kwargs["reduced"] if "reduced" in kwargs else False,
            )

        experiment.reset()
        game = self.game_template(callback=experiment, **self.params)
        network = self.network_template(callback=experiment, game=game, **self.params)

        for e in range(self.params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(self.params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                if hasattr(network, 'reward') and callable(getattr(network, 'reward')):
                    reward = network.reward(state, action)

                if done:
                    break

        experiment.training_end()

        return [*experiment]
