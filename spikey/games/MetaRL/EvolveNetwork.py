"""
Evolve a spiking neural network to learn a RL enviornment.
"""
try:
    from template import MetaRL
except ImportError:
    from spikey.games.MetaRL.template import MetaRL

from spikey.meta import Series
from spikey.meta.backends.single import SingleProcessBackend
from spikey.logging import log


def default_aggregate_fitness(metarl, tracking):
    return sum(tracking) / metarl._n_reruns


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
    training_loop: TrainingLoop
        Configured training loop used in experiments.
    genotype_constraints: dict
        Constraints of genotypes used to train network.
    static_updates: dict {key: [value per rerun]}, default=None
        Updates to a specific network or game parameter. See spikey.meta.Series _static_updates_.
    tracking_getter: lambda network, game, results, info: object
        Get specific value from network, game, results or info after
        training in order to set fitness.
    aggregate_fitness: lambda genotype_fitnesses: float, default=default_aggregate_fitness
        Function to aggregate fitnesses over reruns of same genotype.
    n_reruns: int, default=5
        Times to rerun same genotype, reruns are aggregated into a single fitness
        using the aggregate_fitness function.
    eval_steps: int, default=max
        Number of recent steps to evaluate network performance over.
    win_fitness: float
        Fitness necessary to terminate metarl.

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

    GENOTYPE_CONSTRAINTS = {}  ## NOTE: +1 for all randint

    def __init__(
        self,
        training_loop: object,
        genotype_constraints: dict,
        tracking_getter: callable,
        win_fitness: float,
        static_updates: list = None,
        aggregate_fitness: callable = None,
        n_reruns: int = 5,
        eval_steps: int = None,
    ):
        self.training_loop = training_loop

        self.GENOTYPE_CONSTRAINTS = genotype_constraints
        self.static_updates = static_updates

        self.win_fitness = win_fitness
        self.eval_steps = eval_steps
        self._n_reruns = n_reruns

        self.tracking_getter = tracking_getter
        self.aggregate_fitness = aggregate_fitness or default_aggregate_fitness

        super().__init__()

    def get_fitness(
        self,
        genotype: dict,
        logging: bool = True,
        **kwargs,
    ) -> (float, bool):
        """
        Train a neural network on an RL environment to gauge its fitness.

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
        terminate = False

        DESIRED_INFO = []

        ## Run Simulation
        tracking = []
        run_info = {key: [] for key in DESIRED_INFO}

        params = {
            "eval_steps": self.eval_steps,
            **genotype,
        }

        if self.static_updates is None:
            training_loop = self.training_loop.copy()
            training_loop.reset(params=params)
            series = (training_loop for _ in range(self._n_reruns))

        else:
            training_loop = self.training_loop.copy()
            training_loop.reset(params=params)
            series = Series(
                training_loop,
                self.static_updates,
                backend=SingleProcessBackend(),
            )

        for experiment in series:
            network, game, results, info = experiment(**kwargs)

            tracking.append(self.tracking_getter(network, game, results, info))

            for key in DESIRED_INFO:
                try:
                    run_info[key].append(results[key])
                except KeyError:
                    run_info[key].append(info[key])

        ## Evaluate
        fitness = self.aggregate_fitness(self, tracking)

        if fitness >= self.win_fitness:
            terminate = True

        ## Log
        results.update(
            {
                "n_reruns": self._n_reruns,
                "fitness": fitness,
            }
        )
        info.update({"run_" + key: value for key, value in run_info.items()})

        if logging:
            log_fn = log_kwargs["log_fn"] if "log_fn" in kwargs else log
            log_fn(network, game, results, info, **kwargs)

        return fitness, terminate
