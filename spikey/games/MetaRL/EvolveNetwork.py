"""
Evolve a spiking neural network to learn a RL enviornment.
"""
import numpy as np
from spikey.module import Key
from spikey.games.MetaRL.template import MetaRL
from spikey.meta import Series
from spikey.meta.backends.single import SingleProcessBackend
from spikey.logging import log


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
                "tracking_getter",
                "f(net, game, results, info)->float Get fitness from experiment.",
            ),
            Key(
                "aggregate_fitness",
                "f([fitness, ..])->float Aggregate fitnesses of each rerun.",
                default=np.mean,
            ),
        ]
    )
    GENOTYPE_CONSTRAINTS = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        GENOTYPE_CONSTRAINTS = self._genotype_constraints

    def get_fitness(
        self,
        genotype: dict,
        logging: bool = True,
        **kwargs,
    ) -> (float, bool):
        """
        Train a neural network on an RL environment to gauge its fitness.
        NOTE: Logs results and info only from last rerun.

        Parameters
        ----------
        genotype: dict
            Dictionary with values for each key in GENOTYPE_CONSTRAINTS.
        logging: bool, default=True
            Whether or not to log results to file.
        kwargs: dict, default=None
            Logging and experiment logging keyword arguments.

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
            genotype = [{}, ...]
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
            network, game, results, info = experiment(**kwargs)
            tracking.append(self._tracking_getter(network, game, results, info))

        fitness = self._aggregate_fitness(tracking)
        terminate = fitness >= self._win_fitness

        if logging:
            results.update(
                {
                    "n_reruns": self._n_reruns,
                    "fitness": fitness,
                }
            )

            log_fn = log_kwargs["log_fn"] if "log_fn" in kwargs else log
            log_fn(network, game, results, info, **kwargs)

        return fitness, terminate
