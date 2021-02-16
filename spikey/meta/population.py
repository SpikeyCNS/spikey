"""
An evolving population.

Usage
-----
metagame = EvolveFlorian(Network, Game, TrainingLoop, **metagame_config,)
population = Population(*metagame.population_arguments, **pop_config)

while not population.terminated:
    fitness = population.evaluate()

    population.update(fitness)

    print(f"{population.epoch} - Max fitness: {max(fitness)}")
"""
import os
from copy import copy, deepcopy
import numpy as np
from spikey.module import Module, Key
from spikey.logging import log, MultiLogger
from spikey.meta.backends.default import MultiprocessBackend


class GenotypeMapping:
    """
    Cache genotype-fitness matchings.

    Parameters
    ----------
    n_storing: int
        Number of genotypes to store

    Usage
    -----
    ```python
    cache = GenotypeCache(256)

    cache.update({'a': 1}, 24)

    fitness = cache[{'a': 1}]
    print(fitness)  # -> 24
    ```
    """

    def __init__(self, n_storing: int):
        self.n_storing = n_storing

        self.genotypes = []
        self.fitnesses = []

    def __getitem__(self, genotype: dict) -> float:
        """
        Pull value for specific genotype from cache.

        Parameters
        ----------
        genotype: dict
            Genotype to pull cached value of.

        Returns
        -------
        float or None The cached fitness of the genotype or None.

        Usage
        -----
        ```python
        cache = GenotypeCache(256)

        cache.update({'a': 1}, 24)

        fitness = cache[{'a': 1}]
        print(fitness)  # -> 24
        ```
        """
        genotype_no_age = copy(genotype)
        if "_age" in genotype_no_age:
            del genotype_no_age["_age"]

        if genotype_no_age not in self.genotypes:
            return None

        idx = self.genotypes.index(genotype_no_age)
        fitness = self.fitnesses[idx]

        self.update(genotype, fitness)

        return fitness

    def update(self, genotype: dict, fitness: float):
        """
        Update cache with result.

        Parameters
        ----------
        genotype: dict
            Genotype to use as cache key.
        fitness: float
            Fitness of genotype given.

        Usage
        -----
        ```python
        cache = GenotypeCache(256)

        cache.update({'a': 1}, 24)

        fitness = cache[{'a': 1}]
        print(fitness)  # -> 24
        ```
        """
        if not self.n_storing:
            return

        # shallow copy ok -- only robust to del age in copy
        # mutate, crossover use deepcopy so ok here
        genotype_no_age = copy(genotype)  # deepcopy(genotype)
        if "_age" in genotype_no_age:
            del genotype_no_age["_age"]

        self.genotypes.append(genotype_no_age)
        self.fitnesses.append(fitness)

        assert len(self.genotypes) == len(self.fitnesses), "Cache broken!"

        if len(self.genotypes) >= self.n_storing:
            self.genotypes = self.genotypes[-self.n_storing :]
            self.fitnesses = self.fitnesses[-self.n_storing :]


def run(
    fitness_func: callable, cache: GenotypeMapping, genotype: dict, *params, **kwparams
) -> (float, bool):
    """

    Parameters
    ----------
    fitness_func: callable
        Function to determine fitness of genotype.
    cache: GenotypeMapping
        Genotype-fitness cache.
    genotype: dict
        Genotype to determine fitness of.
    *params, **kwparams: list, dict
        Any parameters necessary for fitness func and logging.

    Returns
    -------
    fitness: float, terminate: bool
    """
    fitness = cache[genotype]
    if fitness is not None:
        if "logging" in kwparams and kwparams["logging"]:
            log(
                None,
                None,
                results={"fitness": fitness},
                info=genotype,
                filename=filename,
            )
        terminate = False

    else:
        fitness, terminate = fitness_func(genotype, *params, **kwparams)

    cache.update(genotype, fitness)

    return fitness, terminate


def checkpoint_population(
    folder: str, file_header: str, epoch: int, genotypes: list, fitnesses: list
):
    """
    Checkpoint current epoch of population in file.

    Parameters
    ----------
    folder: str
        Folder to store checkpoint file.
    file_header: str
        Filename prefix before epoch number.
    epoch: int
        Number of epoch being saved.
    genotypes: list
        Genotypes present in the epoch being checkpointed.
    fitnesses: list
        Fitnesses of genotypes present in epoch being checkpointed.
    """
    from pickle import dump as pickledump

    filename = f"{file_header}~EPOCH-({epoch:03d}).obj"

    with open(os.path.join(folder, filename), "wb") as file:
        pickledump({"genotypes": genotypes, "fitnesses": fitnesses}, file)


def read_population(population: object, folder: str) -> list:
    """
    Read genotypes & fitnesses from last epoch and use it.

    Updates population, returns previous fitnesses

    Parameters
    ----------
    population: Population
        Object to get up to speed with checkpoint.
    folder: path
        Folder to find most recent checkpoint from.

    Returns
    -------
    list Fitnesses of most recent epoch.
    """
    from pickle import load as pickleload

    relevant_filenames = []

    for filename in os.listdir(folder):
        if "EPOCH" in filename:
            relevant_filenames.append(filename)

    if not relevant_filenames:
        raise ValueError(f"Could not find an previous EPOCH data in {folder}!")

    relevant_filenames.sort()

    population.epoch = int(relevant_filenames[-1].split("(")[-1].split(")")[0])

    with open(os.path.join(folder, relevant_filenames[-1]), "rb") as file:
        data = pickleload(file)

    population.population = data["genotypes"]

    return data["fitnesses"]


class Population(Module):
    """
    An evolving population.

    Parameters
    ----------
    genotype_constraints: dict, {str: list/tuple}
        Constraints for each gene, list denotes use random.choice, tuple is use random.uniform.
    get_fitness: func[genotype]->float
        Function to get the fitness of each genotype.
    log_info: dict, default=None
        Parameters for logger.
    folder: str, default="log"
        Folder to save logs in.
    logging: bool, default=False
        Whether to log or not.
    reduced_logging: bool, default=True
        Whether to reduce amount of logging or not.
    backend: object, default=MultiprocessBackend,
        Setup to execute functions in distributed way.
    kwargs: dict, default=None
        Any configuration, required keys listed in NECESSARY_KEYS.

    Usage
    -----
    metagame = EvolveFlorian(Network, Game, TrainingLoop, **metagame_config,)
    population = Population(*metagame.population_arguments, **pop_config)

    while not population.terminated:
        fitness = population.evaluate()

        population.update(fitness)

        print(f"{population.epoch} - Max fitness: {max(fitness)}")
    """

    NECESSARY_KEYS = [
        Key("n_process", "Number of processes to run concurrently.", int),
        Key("n_storing", "Number of genotypes to store in cache.", int),
        Key(
            "n_agents", "Number of agents in population per epoch.", (int, list, tuple)
        ),
        Key("n_epoch", "Number of epochs -- unused if n_agents is iterable.", int),
        Key(
            "mutate_eligable_pct",
            "(0, 1] Pct of prev agents eligable to be mutated.",
            float,
        ),
        Key(
            "max_age",
            "Max age agent can reach before being removed from mutation/crossover/survivor pools.",
            int,
        ),
        Key(
            "random_rate",
            "(0, 1) Percent agents in population to generate randomly.",
            float,
        ),
        Key(
            "survivor_rate",
            "(0, 1) Percent(new generation) previous generation preserved/turn.",
            float,
        ),
        Key(
            "mutation_rate",
            "(0, 1) Percent(new generation) previous generation mutated/turn.",
            float,
        ),
        Key(
            "crossover_rate",
            "(0, 1) Percent(new generation) previous generation crossed over/turn.",
            float,
        ),
    ]

    def __init__(
        self,
        genotype_constraints: dict,
        get_fitness: callable,
        log_info: dict = None,
        folder: str = "log",
        logging: bool = False,
        reduced_logging: bool = True,
        backend: object = None,
        **config,
    ):
        super().__init__(**config)

        self.backend = backend or MultiprocessBackend(config["n_process"])

        # N Agents per epoch
        if isinstance(self._n_agents, (list, tuple, np.ndarray)):
            self.n_agents = (value for value in self._n_agents)
        else:
            self.n_agents = (self._n_agents for _ in range(self._n_epoch))

        self.genotype_constraints = genotype_constraints

        ## Setup
        self.cache = GenotypeMapping(self._n_storing)
        self.population = [self._random() for _ in range(next(self.n_agents))]

        self.epoch = 0  # For summaries
        self.terminated = False

        self.get_fitness = get_fitness

        ##
        if self._mutate_eligable_pct == 0:
            raise ValueError("mutate_eligable pct cannot be 0!")

        ## Normalize update rates to 1.
        self._normalize_rates()

        ## Logging
        self.logging = logging
        self.reduced_logging = reduced_logging

        if self.logging:
            self._setup_logging(folder, log_info, **config)

    def _normalize_rates(self):
        """
        Normalize pertinent algorithm rates to 1.
        """
        total = (
            self._random_rate
            + self._survivor_rate
            + self._mutation_rate
            + self._crossover_rate
        )

        if not total:
            raise ValueError(
                "Need nonzero value for the survivor, mutation or crossover rate."
            )

        self._random_rate /= total
        self._survivor_rate /= total
        self._mutation_rate /= total
        self._crossover_rate /= total

    def _setup_logging(self, folder: str, log_info: dict, **config):
        self.multilogger = MultiLogger(folder=folder)

        info = {"population_config": config}
        if log_info:
            info.update({"metagame_info": log_info})

        self.multilogger.summarize(results=None, info=info)

    def __len__(self) -> int:
        return len(self.population)

    def _genotype_dist(self, genotype1: dict, genotype2: dict) -> float:
        """
        Testing Population._genotype_dist.

        Parameters
        ----------
        genotype1: genotype
            Genotypes to find the distance between.
        genotype2: genotype
            Genotypes to find the distance between.

        Returns
        -------
        Euclidean distance between the two genotypes.
        """
        total = 0

        for key in self.genotype_constraints.keys():
            if isinstance(genotype1[key], (list, tuple)):
                for i in range(len(genotype1[key])):
                    total += (genotype1[key][i] - genotype2[key][i]) ** 2

                continue

            total += (genotype1[key] - genotype2[key]) ** 2

        return total ** 0.5

    def _random(self) -> dict:
        """
        Randomly generate a genotype given constraints.
        """
        eval_constraint = (
            lambda cons: np.random.uniform(*cons)
            if isinstance(cons, tuple)
            else cons[np.random.choice(len(cons))]
        )

        genotype = {
            key: eval_constraint(constraint)
            for key, constraint in self.genotype_constraints.items()
        }

        genotype["_age"] = 0

        return genotype

    def _mutate(self, genotypes: list) -> list:
        """
        Mutate a random key of each genotype given.
        """
        if not isinstance(genotypes, (list, np.ndarray)):
            genotypes = [genotypes]

        new_genotypes = []

        for genotype in genotypes:
            new_genotype = deepcopy(genotype)  ## prevent edit of original!

            key = np.random.choice(list(self.genotype_constraints.keys()))

            cons = self.genotype_constraints[key]

            if isinstance(cons, tuple):
                new_genotype[key] = np.random.uniform(*cons)
            else:
                new_genotype[key] = cons[np.random.choice(len(cons))]

            new_genotype["_age"] = 0

            new_genotypes.append(new_genotype)

        return new_genotypes

    def _crossover(self, genotype1: dict, genotype2: dict) -> [dict, dict]:
        """
        Crossover two different genotypes.

        Parameters
        ----------
        genotype: dict, str: float
            Genotype.

        Returns
        -------
        2 new genotypes.
        """
        offspring1, offspring2 = {}, {}

        switch = False
        switch_key = np.random.choice(list(self.genotype_constraints.keys()))

        keys = list(self.genotype_constraints.keys())
        np.random.shuffle(keys)  # Prevent bias

        for key in keys:
            if key == switch_key:
                switch = True

            offspring1[key] = genotype1[key] if switch else genotype2[key]
            offspring2[key] = genotype2[key] if switch else genotype1[key]

        offspring1["_age"] = 0
        offspring2["_age"] = 0

        return [offspring1, offspring2]

    def update(self, f: list):
        """
        Update the population based on each agents fitness.

        Parameters
        ----------
        f: list of float
            Fitness values for each agent.

        Effects
        -------
        Population will updated according to operator rates.
        """
        try:
            n_agents = next(self.n_agents)
        except StopIteration:
            self.terminated = True
            return

        prev_gen = [(self.population[i], f[i]) for i in range(len(f))]
        prev_gen = sorted(prev_gen, key=lambda x: x[1])
        prev_gen = [value[0] for value in prev_gen if value[0]["_age"] < self._max_age]

        self.population = []

        ## Generate Random
        self.population += [
            self._random() for _ in range(int(n_agents * self._random_rate))
        ]

        ## Choose Survivors -- Elitist, will not survive if too old
        if int(n_agents * self._survivor_rate):  # -0 returns whole list!!
            survivors = [
                deepcopy(genotype)
                for genotype in prev_gen[-int(n_agents * self._survivor_rate) :]
            ]

            for genotype in survivors:
                genotype["_age"] += 1

            self.population += survivors

        ## Mutate
        mutate_candidates = prev_gen[-int(self._mutate_eligable_pct * len(prev_gen)) :]
        self.population += self._mutate(
            [
                deepcopy(genotype)
                for genotype in np.random.choice(
                    mutate_candidates, size=int(n_agents * self._mutation_rate)
                )
            ]
        )

        ## Crossover
        for _ in range(int(n_agents * self._crossover_rate) // 2):
            genotype1 = np.random.choice(prev_gen)
            genotype2 = np.random.choice(prev_gen)

            self.population += self._crossover(deepcopy(genotype1), deepcopy(genotype2))

        ## Ensure correct n_agents
        if len(self) < n_agents:
            diff = n_agents - len(self)

            self.population += self._mutate(np.random.choice(prev_gen, size=diff))

    def evaluate(self) -> list:
        """
        Evaluate each agent on the fitness function.

        Returns
        -------
        Fitness values for each agent.
        """
        fitnesses = []
        self.epoch += 1

        params = [
            (
                self.get_fitness,
                self.cache,
                genotype,
                log if self.logging else None,
                next(self.multilogger.filename_generator) if self.logging else None,
                self.reduced_logging,
            )
            for genotype in self.population
        ]

        results = self.backend.distribute(run, params)

        fitnesses = [result[0] for result in results]
        if any([result[1] for result in results]):
            self.terminated = True

        if self.logging:
            checkpoint_population(
                folder="",
                file_header=self.multilogger.prefix,
                epoch=self.epoch,
                genotypes=self.population,
                fitnesses=fitnesses,
            )

        return fitnesses
