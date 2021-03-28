"""
Pre-built, reusable training loops that allow users
to train a network on a game individually, repeatedly
or within the meta analysis tools.
"""
from copy import deepcopy

from spikey.module import Module, Key
from spikey.core.callback import ExperimentCallback, RLCallback


class TrainingLoop(Module):
    """
    Template for pre-built, reusable training loops.

    Parameters
    ----------
    network_template: Network[type] or Network
        Network to train.
    game_template: RL[type] or RL
        Game to train.
    callback: ExperimentCallback[class or object], default=ExperimentCallback
        Callback object template or already initialized callback.
    **params: dict
        Network, game and training parameters.

    Examples
    --------

    .. code-block:: python

        experiment = TrainingLoop(Network(**config), RL(**config), **config)
        experiment.reset()

        network, game, results, info = experiment()

    .. code-block:: python

        experiment = TrainingLoop(Network, RL, RLCallback **config)
        experiment.reset()

        network, game, results, info = experiment()

    .. code-block:: python

        callback = RLCallback
        experiment = TrainingLoop(Network, RL, callback **config)
        experiment.reset()

        network, game, results, info = experiment()
    """

    NECESSARY_KEYS = []

    def __init__(
        self,
        network_template: type,
        game_template: type,
        callback: ExperimentCallback = None,
        **params: dict,
    ):
        self.network_template = network_template
        self.game_template = game_template

        if not len(params):
            print(f"WARNING: No values given as {type(self)} params!")
        self.params = {}
        if hasattr(self.network_template, "keys"):
            self.params.update(deepcopy(self.network_template.keys))
        self.params.update(params)

        super().__init__(**self.params)
        self.callback = self._init_callback([callback, RLCallback][callback is None])

    def reset(
        self,
        network_template: type = None,
        game_template: type = None,
        callback: ExperimentCallback = None,
        params: dict = None,
        **kwparams: dict,
    ):
        """
        Reset, optionally override network_template, game_template and parameters.

        Parameters
        ----------
        network_template: Network[type] or Network, default=None
            Network to train.
        game_template: RL[type] or RL, default=None
            Game to train.
        callback: ExperimentCallback[class or object], default=None
            Callback object template or already initialized callback.
        **params: dict
            Updates to network, game and training parameters.

        Examples
        --------

        .. code-block:: python

            experiment = TrainingLoop(Network, RL, **config)
            experiment.reset()
        """
        if network_template is not None:
            self.network_template = network_template
        if game_template is not None:
            self.game_template = game_template
        if params is not None:
            self.params.update(params)
        if kwparams is not None:
            self.params.update(kwparams)
        if callback is not None:
            self.callback = self._init_callback(callback)

        self.callback.reset()

    def bind(self, name):
        """
        Add binding for monitors and later referencing.
        All bindings must be set before calling reset.
        """
        self.callback.bind(name)

    def monitor(
        self,
        function: str,
        location: str,
        identifier: str,
        target: list,
        method: str = "list",
    ):
        """
        Setup runtime monitoring for a new parameter on the callback.

        Parameters
        ----------
        function: str
            Name of callback to attach to.
        location: str
            Storage location, 'results' or 'info'.
        identifier: str
            Key to save target in.
        target: list[str]
            Location of information, eg ['network', 'synapse', 'spike_log'].
            arg, arg_<int> are reserved for accessing kwargs and list[<int>] respectively.
        method: 'scalar' or 'list'
            Monitoring method, whether to store as list or scalar.

        Examples
        --------

        .. code-block:: python

            training_loop.monitor('training_end', 'results', 'processing_time', ['network', 'processing_time'], 'scalar')

        .. code-block:: python

            training_loop.monitor('network_tick', 'info', 'step_actions', ['arg_1'], 'list')
        """
        self.callback.monitor(function, location, identifier, target, method)

    def log(self, **log_kwargs):
        """
        Log data to file via callback.log.

        Parameters
        ----------
        log_kwargs: dict
            Log function kwargs
        """
        self.callback.log(**log_kwargs)

    def _init_callback(self, callback):
        """
        Initialize callback object for TrainingLoop.
        """
        if type(callback) == type:
            callback = callback(**self.params)
        return callback

    def init(self) -> (object, object):
        """
        Setup callback and initialize network and game.
        """
        self.callback.reset(
            **{
                key.name if isinstance(key, Key) else key: self.params[key]
                for key in self.NECESSARY_KEYS
            }
        )
        if callable(self.game_template):
            game = self.game_template(callback=self.callback, **self.params)
        else:
            game = self.game_template
            game.callback = self.callback
            self.callback.game_init(game)
        if callable(self.network_template):
            network = self.network_template(
                callback=self.callback, game=game, **self.params
            )
        else:
            network = self.network_template
            network.callback = self.callback
            self.callback.network_init(network)

        if self.training:
            if network is not None:
                network.train()
            if game is not None:
                game.train()
        else:
            if network is not None:
                network.eval()
            if game is not None:
                game.eval()

        return network, game

    def __call__(self) -> (object, object, dict, dict):
        """
        Run training loop a single time.

        Returns
        -------
        network: Network, game: RL, results: dict, info: dict.

        Examples
        --------

        .. code-block:: python

            experiment = TrainingLoop(Network, RL, **config)
            experiment.reset()

            network, game, results, info = experiment()
        """
        raise NotImplementedError(f"Call not implemented in {type(self)}.")


class GenericLoop(TrainingLoop):
    """
    Generic reinforcement learning training loop.

    .. code-block:: python

        for ep in n_episodes:
            while not done or until i == len_episode:
                action = network.tick(state)
                state_next, _, done, __ = game.step(action)
                reward = network.reward(state, action)
                state = state_next

    Parameters
    ----------
    network_template: Network[type] or Network
        Network to train.
    game_template: RL[type] or RL
        Game to train.
    params: dict
        Network, game and training parameters.

    Examples
    --------

    .. code-block:: python

        experiment = GenericLoop(Network, RL, **config)
        experiment.reset()

        network, game, results, info = experiment()
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

        Examples
        --------

        .. code-block:: python

            experiment = TrainingLoop(Network, RL, **config)
            experiment.reset()

            network, game, results, info = experiment()
        """
        network, game = self.init()

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

        self.callback.training_end()

        return [*self.callback]
