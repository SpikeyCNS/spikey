"""
Base meta reinforcement learning environment template.
"""
from spikey.module import Module
from queue import Queue


class MetaRL(Module):
    """
    Base meta reinforcement learning environment template.

    Parameters
    ----------
    preset: str=PRESETS.keys(), default=None
        Configuration preset key, default values for game parameters.
    kwargs: dict, default=None
        Game parameters for CONFIG_DESCRIPTIONS. Overrides preset settings.

    Usage
    -----
    ```python
    metagame = MetaRL()
    game.seed(0)

    for _ in range(100):
        genotype = [{}, ...]
        fitness, done = metagame.get_fitness(genotype)

        if done:
            break

    game.close()
    ```

    ```python
    metagame = MetaRL(**metagame_config)
    game.seed(0)

    population = Population(... metagame, ...)
    # population main loop
    ```
    """

    CONFIG_DESCRIPTIONS = {}
    GENOTYPE_CONSTRAINTS = {}
    PRESETS = {}

    def __init__(self, preset: str = None, **kwargs):
        self._params = {}

        if preset is not None:
            self._params.update(self.PRESETS[preset])
        if hasattr(self, "config"):
            self._params.update(self.config)

        self._params.update(
            {key: kwargs[key] for key in self.CONFIG_DESCRIPTIONS if key in kwargs}
        )

        super().__init__(**self._params)

    @property
    def params(self) -> dict:
        """
        Configuration of game.
        """
        return self._params

    @property
    def population_arguments(self) -> (dict, callable):
        """
        Easily accessible game params helpful for meta tool initialization.

        Returns
        -------
        GENOTYPE_CONSTRAINTS: dict, get_fitness: callable

        Usage
        -----
        ```python
        metagame = MetaRL(**metagame_config)
        population = Population(*metagame.population_arguments)
        ```
        """
        return self.GENOTYPE_CONSTRAINTS, self.get_fitness

    def get_fitness(
        self,
        genotype: dict,
        log: callable = None,
        filename: str = None,
        reduced_logging: bool = True,
        q: Queue = None,
    ) -> (float, bool):
        """
        Evaluate the fitness of a genotype.

        Parameters
        ----------
        genotype: dict
            Dictionary with values for each key in GENOTYPE_CONSTRAINTS.
        log: callable, default=None
            log function: (network, game, results, info, filename=filename).
        filename: str, default=None
            Filename for logging function.
        reduced_logging: bool, default=True
            Whether to reduce amount of logging from this function or not.
        q: Queue, default=None
            Queue to append (genotype, fitness, terminate).

        Returns
        -------
        fitness: float
            Fitness of genotype given.
        done: bool
            Whether termination condition has been reached or not.

        Usage
        -----
        ```python
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
        raise NotImplementedError(f"get_fitness not implemented for {type(self)}!")
        return 0, False
