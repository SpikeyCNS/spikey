"""
Base meta reinforcement learning environment template for use
with meta/population.
"""
from spikey.games.game import Game


class MetaRL(Game):
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
        super().__init__(preset, **kwargs)

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
        ```
        """
        fitness, done = self.get_fitness(action, **kwargs)

        info = {}
        return None, fitness, done, info
