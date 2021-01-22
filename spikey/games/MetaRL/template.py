"""
Template for MetaRL games.
"""
from queue import Queue


class MetaRL:
    """
    MetaRL template.
    """

    GENOTYPE_CONSTRAINTS = {}
    CONFIG_DESCRIPTIONS = {}
    PRESETS = {}

    def __init__(self, preset: str = None, **kwargs):
        ## Generate config
        self._params = {}

        if preset is not None:
            self._params.update(self.PRESETS[preset])
        if hasattr(self, "config"):
            self._params.update(self.config)

        self._params.update(
            {key: kwargs[key] for key in self.CONFIG_DESCRIPTIONS if key in kwargs}
        )

    @property
    def params(self) -> dict:
        """
        Configuration of game.
        """
        return self._params

    @property
    def population_arguments(self) -> (dict, callable):
        """
        Easily accessible game params necessary for population initialization.
        """
        return self.GENOTYPE_CONSTRAINTS, self.get_fitness

    def get_fitness(
        self,
        genotype: dict,
        log: callable = None,
        filename: str = None,
        reduced_logging: bool = True,
        q: Queue = None,
    ) -> (list, bool):
        """
        Evaluate a genotype.

        Returns
        -------
        Fitness of genotype, whether the termination condition has been reached.
        """
        raise NotImplementedError(f"get_fitness not implemented for {type(self)}!")
        return [], False
