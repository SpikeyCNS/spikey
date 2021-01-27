"""
Evolve a spiking neural network to learn a RL enviornment.
"""
try:
    from template import MetaRL
except ImportError:
    from spikey.games.MetaRL.template import MetaRL

from spikey.meta import Series
from spikey.meta.backends.single import SingleProcessBackend


def default_aggregate_fitness(metarl, tracking):
    return sum(tracking) / metarl._n_reruns


class EvolveNetwork(MetaRL):
    """
    An enviornment to evolve a spiking neural network on a standard RL enviornment.

    Parameters
    ----------
    n_episodes: int
        Number of episodes to run.
    len_episode: int
        Max length of episode.
    win_fitness: float
        Fitness needed for win condition.
    network_template: SNN
        Template of network to use.
    game_template: RL
        Template of game to use.
    training_loop: TrainingLoop
        Training loop to train network in game.
    genotype_constraints: dict
        Constraints on genotypes - () = floating point, [] = random choice
    static_config: dict
        Part of config that is static to all runs.
    tracking_getter: f(network, game, results, info) -> Any
        Pull data from each experiment to track.
    aggregate_fitness: f(list[tracking_getter()]) -> fitness
        Calculate fitness over multiple experiment runs.
    n_reruns: int, default=5 -- overloaded with len(static_updates it's if not None)
        Number of times to rerun and average fitness.

    static_udpdates: List[Dict], default=None -- overloads n_reruns it's if not None
        Iterable of parameters to update static_config with before each run.
    """

    STATIC_CONFIG = {}
    GENOTYPE_CONSTRAINTS = {}  ## NOTE: +1 for all randint

    def __init__(
        self,
        n_episodes,
        len_episode,
        win_fitness,
        network_template,
        game_template,
        training_loop,
        genotype_constraints,
        static_config,
        tracking_getter,
        aggregate_fitness=None,
        n_reruns=5,
        static_updates=None,
        eval_steps=None,
    ):
        self.n_episodes = n_episodes
        self.len_episode = len_episode
        self.win_fitness = win_fitness
        self.eval_steps = eval_steps
        self._n_reruns = n_reruns

        self.tracking_getter = tracking_getter
        self.aggregate_fitness = aggregate_fitness or default_aggregate_fitness

        self.training_loop = training_loop

        self.network_template = network_template
        self.game_template = game_template

        self.GENOTYPE_CONSTRAINTS = genotype_constraints
        self.STATIC_CONFIG = static_config

        self.static_updates = static_updates

        super().__init__()

    def get_fitness(
        self, genotype, log=None, filename=None, reduced_logging=True, q=None
    ):
        """
        Train a neural network and return its fitness.
        """
        terminate = False

        DESIRED_INFO = []

        ## Run Simulation
        tracking = []
        run_info = {key: [] for key in DESIRED_INFO}

        params = {
            "n_episodes": self.n_episodes,
            "len_episode": self.len_episode,
            "eval_steps": self.eval_steps,
            **self.STATIC_CONFIG,
            **genotype,
        }

        if self.static_updates is None:
            training_loop = self.training_loop(
                self.network_template, self.game_template, params
            )
            series = (training_loop for _ in range(self._n_reruns))

        else:
            series = Series(
                self.training_loop,
                self.network_template,
                self.game_template,
                params,
                self.static_updates,
                backend=SingleProcessBackend(),
            )

        for experiment in series:
            network, game, results, info = experiment(reduced_logging=reduced_logging)

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

        if q is not None:
            q.put((genotype, fitness, terminate))

        ## Log
        results.update(
            {
                "n_reruns": self._n_reruns,
                "fitness": fitness,
            }
        )
        info.update({"run_" + key: value for key, value in run_info.items()})

        if log is not None:
            if filename is None:
                raise ValueError("Filename must have value if logging enabled!")

            log(network, game, results, info, filename=filename)

        return fitness, terminate
