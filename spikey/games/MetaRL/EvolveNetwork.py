"""
Evolve a spiking neural network to learn a RL enviornment for use
with meta/population.
"""
import numpy as np
from spikey.module import Key
from spikey.games.MetaRL.template import MetaRL
from spikey.meta import Series
from spikey.meta.backends.single import SingleProcessBackend


class EvolveNetwork(MetaRL):
    """
    An environment to tune spiking neural network parameters on a RL game.

    GENOTYPE_CONSTRAINTS
    --------------------
    Parameterized with the genotype_constraints init parameter.
    Networks are parameterized with a combination of their genotyp and
    original config with the genotype taking priority.

    Parameters
    ----------
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.

    Usage
    -----
    ```python
    metagame = EvolveNetwork()
    game.seed(0)

    for _ in range(100):
        genotype = [{}, ...]
        fitness, done = metagame.get_fitness(genotype)

        if done:
            break

    game.close()
    ```

    ```python
    metagame = EvolveNetwork(**metagame_config)
    game.seed(0)

    population = Population(... metagame, ...)
    # population main loop
    ```
    """

    NECESSARY_KEYS = MetaRL.extend_keys(
        [
            Key("training_loop", "Pre-configured trainingloop used in experiments."),
            Key(
                "genotype_constraints",
                "Constraints of genotypes (training_loop parameters).",
                dict,
            ),
            Key(
                "static_updates",
                "Updates to a specific network or game parameter. See spikey.meta.Series _static_updates_.",
                default=None,
            ),
            Key("n_reruns", "Number of times to rerun experiment", int, default=2),
            Key("win_fitness", "Fitness necessary to terminate MetaRL.", float),
            Key(
                "fitness_getter",
                "f(net, game, results, info)->float Get fitness from experiment.",
            ),
            Key(
                "fitness_aggregator",
                "f([fitness, ..])->float Aggregate fitnesses of each rerun.",
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

        Usage
        -----
        ```python
        metagame = EvolveNetwork()
        game.seed(0)

        for _ in range(100):
            genotype = {}
            fitness, done = metagame.get_fitness(genotype)

            if done:
                break

        game.close()
        ```
        """
        training_loop = self._training_loop.copy()
        training_loop.reset(params=genotype)
        series = Series(
            training_loop,
            self._static_updates,
            backend=SingleProcessBackend(),
        )

        tracking = []
        for experiment in series:
            network, game, results, info = experiment(**self.params)
            tracking.append(self._fitness_getter(network, game, results, info))

        fitness = self._fitness_aggregator(tracking)
        terminate = fitness >= self._win_fitness

        return fitness, terminate
