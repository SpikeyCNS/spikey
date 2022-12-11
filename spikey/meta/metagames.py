"""
Base meta reinforcement learning environment template for use with meta/population.

MetaNQueens is a game to try and place a number of queen chess pieces on a chess
board without any of them being to attack another in the same move. Meant as a
benchmark for Population.

EvolveNetwork is an environment to tune spiking neural network parameters on an RL game.
"""
from spikey.module import Module, Key
import numpy as np
from spikey.meta.backends.single import SingleProcessBackend


class MetaRL(Module):
    """
    Base meta reinforcement learning environment template.

    Parameters
    ----------
    preset: str=PRESETS.keys(), default=None
        Configuration preset key, default values for game parameters.
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.

    Examples
    --------

    .. code-block:: python

        metagame = MetaRL()
        game.seed(0)
        for _ in range(100):
            genotype = [{}, ...]
            fitness, done = metagame.get_fitness(genotype)
            if done:
                break
        game.close()

    .. code-block:: python

        metagame = MetaRL(**metagame_config)
        game.seed(0)
        population = Population(... metagame, ...)
        # population main loop
    """

    NECESSARY_KEYS = []
    GENOTYPE_CONSTRAINTS = {}
    PRESETS = {}

    def __init__(self, preset: str = None, **kwargs):
        self._params = {}
        if preset is not None:
            self._params.update(self.PRESETS[preset])
        if hasattr(self, "config"):
            self._params.update(self.config)
        self._params.update(
            {
                key.name if hasattr(key, "name") else key: kwargs[key]
                for key in self.NECESSARY_KEYS
                if key in kwargs
            }
        )

        super().__init__(**self._params)
        self._add_values(self._params, dest=self._params, prefix="")

    @property
    def params(self) -> dict:
        """
        Configuration of game.
        """
        return self._params

    def reset(self):
        """
        Reset the game, typically unused for MetaRL.
        """

    def get_fitness(self, genotype: dict) -> (float, bool):
        """
        Evaluate the fitness of a genotype.

        Parameters
        ----------
        genotype: dict
            Dictionary with values for each key in GENOTYPE_CONSTRAINTS.

        Returns
        -------
        fitness: float
            Fitness of genotype given.
        done: bool
            Whether termination condition has been reached or not.

        Examples
        --------

        .. code-block:: python

            metagame = MetaRL()
            game.seed(0)
            for _ in range(100):
                genotype = [{}, ...]
                fitness, done = metagame.get_fitness(genotype)
                if done:
                    break
            game.close()
        """
        raise NotImplementedError(f"get_fitness not implemented for {type(self)}!")
        return 0, False

    def step(self, action: dict, **kwargs) -> (object, float, bool, dict):
        """
        Act within the environment.
        gym.Env friendly alias for MetaRL.get_fitness.

        Parameters
        ----------
        action: dict
            Genotype chosen.
        kwargs: dict, default=None
            Optional arguments for get_fitness.

        Returns
        -------
        state: None
            Current state of environment.
        reward: float
            Reward given by environment.
        done: bool
            Whether the game is done or not.
        info: dict
            Information of environment.

        Examples
        --------

        .. code-block:: python

            metagame = MetaRL()
            game.seed(0)
            for _ in range(100):
                genotype = [{}, ...]
                fitness, done = metagame.get_fitness(genotype)
                if done:
                    break
            game.close()
        """
        fitness, done = self.get_fitness(action, **kwargs)

        info = {}
        return None, fitness, done, info

    def close(self):
        """
        Shut down environment.

        Examples
        --------

        .. code-block:: python

            game = Game()
            state = game.reset()

            # training loop

            game.close()
        """
        pass

    def seed(self, seed: int = None):
        """
        Seed random number generators for environment.
        """
        if seed:
            np.random.seed(seed)

        return np.random.get_state()


class MetaNQueens(MetaRL):
    """
    Game to try and place a number of queen chess pieces on a chess
    board without any of them being to attack another in the same move.

    92 distinct solutions out of 4 billion possibilities w/ 8 queens.

    Genotypes are parameterized as follows,

    .. code-block:: python

        for i in range(n_agents):
            xi: int in {0, 7} X position of queen i.
            yi: int in {0, 7} Y position of queen i.

    Parameters
    ----------
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.

    Examples
    --------

    .. code-block:: python

        metagame = MetaNQueens()
        game.seed(0)
        for _ in range(100):
            genotype = [{}, ...]
            fitness, done = metagame.get_fitness(genotype)
            if done:
                break
        game.close()

    .. code-block:: python

        metagame = MetaNQueens(**metagame_config)
        game.seed(0)
        population = Population(... metagame, ...)
        # population main loop
    """

    NECESSARY_KEYS = MetaRL.extend_keys(
        [
            Key(
                "n_queens",
                "{1..8}Number of queens agent needs to place on board.",
                int,
                default=8,
            )
        ]
    )
    GENOTYPE_CONSTRAINTS = {}  ## Defined in __init__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._n_queens > 8 or self._n_queens < 1:
            raise ValueError(f"n_queens must be in range [1, 8], not {self._n_queens}!")

        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h"][: self._n_queens]
        keys = [first + second for second in ["x", "y"] for first in self.letters]

        self.GENOTYPE_CONSTRAINTS = {key: list(range(8)) for key in keys}

    @staticmethod
    def setup_game() -> list:
        """
        Setup game.

        Returns
        -------
        list Initial board state, number of queens in each horizontal, vertical and diagonal line.
        """
        horizontals = np.zeros(8)
        verticals = np.zeros(8)
        ldiagonals = np.zeros(15)  # \
        rdiagonals = np.zeros(15)  # /

        return horizontals, verticals, ldiagonals, rdiagonals

    @staticmethod
    def run_move(board: list, move: tuple) -> list:
        """
        Execute action.

        Parameters
        ----------
        board: list
            Number of queens across each horizontal, vertical and diagonal line.
        move: (x, y) in [0, 7]
            X and Y coordinate to place queen.

        Returns
        -------
        [horizontals: list, verticals: list, ldiagonals: list, rdiagonals: list] Updated board.
        """
        horizontals, verticals, ldiagonals, rdiagonals = board
        x, y = move

        horizontals[x] += 1
        verticals[y] += 1
        ldiagonals[x + y] += 1
        rdiagonals[7 - x + y] += 1

        return horizontals, verticals, ldiagonals, rdiagonals

    def get_fitness(
        self,
        genotype: dict,
    ) -> (float, bool):
        """
        Evaluate the fitness of a genotype.

        Parameters
        ----------
        genotype: dict
            Dictionary with values for each key in GENOTYPE_CONSTRAINTS.

        Returns
        -------
        fitness: float
            Fitness of genotype given.
        done: bool
            Whether termination condition has been reached or not.

        Examples
        --------

        .. code-block:: python

            metagame = MetaNQueens()
            game.seed(0)
            for _ in range(100):
                genotype = [{}, ...]
                fitness, done = metagame.get_fitness(genotype)
                if done:
                    break
            game.close()
        """
        board = self.setup_game()

        for letter in self.letters:
            move = (genotype[letter + "x"], genotype[letter + "y"])

            board = self.run_move(board, move)

        clashes = 0

        for item in board:
            clashes += np.sum(item[item > 1] - 1)

        fitness = 28 - clashes
        terminate = clashes == 0

        return fitness, terminate


class EvolveNetwork(MetaRL):
    """
    An environment to tune spiking neural network parameters on a RL game.

    GENOTYPE_CONSTRAINTS are parameterized by the user with the genotype_constraints init parameter.
    Networks are parameterized with a combination of their genotype and
    original config with the genotype taking priority.

    Parameters
    ----------
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.

    Examples
    --------

    .. code-block:: python

        metagame = EvolveNetwork()
        game.seed(0)
        for _ in range(100):
            genotype = [{}, ...]
            fitness, done = metagame.get_fitness(genotype)
            if done:
                break
        game.close()

    .. code-block:: python

        metagame = EvolveNetwork(**metagame_config)
        game.seed(0)
        population = Population(... metagame, ...)
        # population main loop
    """

    NECESSARY_KEYS = MetaRL.extend_keys(
        [
            Key(
                "training_loop",
                "Pre-configured trainingloop to run and gauge fitness of.",
            ),
            Key(
                "genotype_constraints",
                "A constraint for every trainingloop parameter that should be trained.",
                dict,
            ),
            Key(
                "static_updates",
                "Updates to a specific network or game parameter. ",
                default=None,
            ),
            Key(
                "n_reruns", "Number of times to rerun each experiment.", int, default=2
            ),
            Key(
                "win_fitness", "Fitness threshold necessary to terminate MetaRL.", float
            ),
            Key(
                "fitness_getter",
                "f(net, game, results, info)->float Function to determine experiment fitness.",
            ),
            Key(
                "fitness_aggregator",
                "f([fitness, ..])->float Aggregate fitnesses of each experiment rerun.",
                default=np.mean,
            ),
        ]
    )
    GENOTYPE_CONSTRAINTS = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.GENOTYPE_CONSTRAINTS = self._genotype_constraints

    def get_fitness(self, genotype: dict) -> (float, bool):
        """
        Train a neural network on an RL environment to gauge its fitness.

        Parameters
        ----------
        genotype: dict
            Dictionary with values for each key in GENOTYPE_CONSTRAINTS.

        Returns
        -------
        fitness: float
            Fitness of genotype given.
        done: bool
            Whether termination condition has been reached or not.

        Examples
        --------

        .. code-block:: python

            metagame = EvolveNetwork()
            game.seed(0)
            for _ in range(100):
                genotype = [{}, ...]
                fitness, done = metagame.get_fitness(genotype)
                if done:
                    break
            game.close()
        """
        training_loop = self._training_loop.copy()
        training_loop.reset(**genotype, **self.params)
        series = Series(
            training_loop,
            self._static_updates,
            backend=SingleProcessBackend(),
        )

        tracking = []
        for experiment in series:
            for _ in range(self._n_reruns):
                network, game, results, info = experiment()
                tracking.append(self._fitness_getter(network, game, results, info))

        fitness = self._fitness_aggregator(tracking)
        terminate = fitness >= self._win_fitness

        return fitness, terminate
