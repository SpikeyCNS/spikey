"""
Pre-built, reusable training loops.
"""
from copy import deepcopy

from spikey.core.callback import RLCallback


class TrainingLoop:
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

    NECESSARY_CONFIG = {
        "n_episodes": "int Number of episodes to run,",
        "len_episode": "int Length of episode.",
    }

    def __init__(self, network_template: object, game_template: object, params: dict):
        self.network_template = network_template
        self.game_template = game_template

        self.params = deepcopy(self.network_template.config)
        self.params.update(params)

    def reset(
        self,
        network_template: object = None,
        game_template: object = None,
        params: dict = None,
    ) -> (object, object):
        """
        Reset, optionally override

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

    def __call__(self, **kwargs) -> object:
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

    NECESSARY_KEYS = deepcopy(TrainingLoop.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "n_episodes": "int Number of episodes to run,",
            "len_episode": "int Length of episode.",
        }
    )

    def __call__(self, **kwargs) -> ("SNN", "RL", dict, dict):
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

                reward = network.reward(state, action)

                if done:
                    break

        experiment.training_end()

        return [*experiment]


class TDLoop(TrainingLoop):
    """
    ## Deprecated
    Temporal difference oriented reinforcement learning training loop.

    ```
    for ep in n_episodes:
        while not done or until i == len_episode:
            action = network.tick(state)
            state, _, done, __ = game.step(action)
            network.rewarder.critic_spikes = network.spike_log
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
    experiment = TDLoop(Network, RL, **config)
    experiment.reset()

    network, game, results, info = experiment()
    ```
    """

    NECESSARY_KEYS = deepcopy(TrainingLoop.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "n_episodes": "int Number of episodes to run,",
            "len_episode": "int Length of episode.",
        }
    )

    def __call__(self, **kwargs) -> ("SNN", "RL", dict, dict):
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
        experiment.track(
            "network_reward",
            "info",
            "td_td",
            ["network", "rewarder", "prev_td"],
            "list",
        )
        experiment.track(
            "network_reward",
            "info",
            "td_reward",
            ["network", "rewarder", "prev_reward"],
            "list",
        )
        experiment.track(
            "network_reward",
            "info",
            "td_value",
            ["network", "rewarder", "prev_value"],
            "list",
        )
        experiment.reset()

        game = self.game_template(callback=experiment, **self.params)
        network = self.network_template(callback=experiment, game=game, **self.params)

        try:
            for e in range(self.params["n_episodes"]):
                network.reset()
                state = game.reset()

                network.rewarder.prev_value = None
                network.rewarder.time = 0

                for s in range(self.params["len_episode"]):
                    action = network.tick(state)

                    state, _, done, __ = game.step(action)

                    if network.readout._n_outputs:
                        network.rewarder.critic_spikes = network.spike_log[
                            -network._processing_time :,
                            -network._n_neurons : -network.readout._n_outputs,
                        ]
                    else:
                        network.rewarder.critic_spikes = network.spike_log[
                            -network._processing_time :, -network._n_neurons :
                        ]

                    reward = network.reward(state, action)

                    if done:
                        break

        except KeyboardInterrupt:
            pass

        experiment.training_end()

        return [*experiment]
