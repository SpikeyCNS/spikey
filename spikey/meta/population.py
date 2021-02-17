"""
An evolving population.

Usage
-----
metagame = EvolveFlorian(GenericLoop(network, game, params), **metagame_config,)
population = Population(metagame, **pop_config)

while not population.terminated:
    fitness = population.evaluate()

    population.update(fitness)

    print(f"{population.epoch} - Max fitness: {max(fitness)}")
"""
import os
from copy import copy, deepcopy
import numpy as np
from spikey.module import Module, Key
from spikey.meta.backends.default import MultiprocessBackend
from spikey.logging import log, MultiLogger


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


def run(fitness_func: callable, cache: GenotypeMapping, genotype: dict, log_fn: callable, filename: str) -> (float, bool):
    """

    Parameters
    ----------
    fitness_func: callable
        Function to determine fitness of genotype.
    cache: GenotypeMapping
        Genotype-fitness cache.
    genotype: dict
        Current genotype to test.

    Returns
    -------
    fitness: float, terminate: bool
    """
    fitness = cache[genotype]
    if fitness is not None:
        terminate = False
    else:
        fitness, terminate = fitness_func(genotype)

    if filename:
        results = {
            "fitness": fitness,
            "filename": filename,
        }
        log_fn(
            None,
            None,
            results=results,
            info=genotype,
            filename=filename,
        )

    cache.update(genotype, fitness)

    return fitness, terminate


def checkpoint_population(population: object, folder: str = ""):
    """
    Checkpoint current epoch of population in file.

    Parameters
    ----------
    population: Population
        Population to checkpoint.
    folder: str
        Folder to store checkpoint file.
    """
    from pickle import dump as pickledump

    epoch = population.epoch
    genotypes = population.population

    if hasattr(population, "multilogger"):
        file_header = population.multilogger.prefix
    else:
        file_header = ""
    filename = f"{file_header}~EPOCH-({epoch:03d}).obj"

    with open(os.path.join(folder, filename), "wb") as file:
        pickledump({"genotypes": genotypes}, file)


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
    game: MetaRL
        MetaRL game to evolve agents for.
    backend: MetaBackend, default=MultiprocessBackend(max_process)
        Backend to execute experiments with.
    max_process: int, default=16
        Number of separate processes to run experiments for
        default backend.
    kwargs: dict, default=None
        Any configuration, required keys listed in NECESSARY_KEYS.

    Usage
    -----
    metagame = EvolveFlorian(GenericLoop(network, game, params), **metagame_config,)
    population = Population(metagame, **pop_config)

    while not population.terminated:
        fitness = population.evaluate()

        population.update(fitness)

        print(f"{population.epoch} - Max fitness: {max(fitness)}")
    """

    NECESSARY_KEYS = [
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
        Key("logging", "Whether to log or not.", bool, default=True),
        Key("log_fn", "f(n, g, r, i, filename) Logging function.", default=log),
        Key("folder", "Folder to save logs to.", str, default="log"),
    ]

    def __init__(
        self,
        game: object,
        backend: object = None,
        max_process: int = 16,
        **config,
    ):
        super().__init__(**config)

        self.genotype_constraints = game.GENOTYPE_CONSTRAINTS
        self.get_fitness = game.get_fitness
        self.backend = backend or MultiprocessBackend(max_process)

        if isinstance(self._n_agents, (list, tuple, np.ndarray)):
            self.n_agents = (value for value in self._n_agents)
        else:
            self.n_agents = (self._n_agents for _ in range(self._n_epoch))

        self.cache = GenotypeMapping(self._n_storing)
        self.population = [self._random() for _ in range(next(self.n_agents))]

        self.epoch = 0  # For summaries
        self.terminated = False

        if self._mutate_eligable_pct == 0:
            raise ValueError("mutate_eligable pct cannot be 0!")

        self._normalize_rates()
        if self._logging:
            self._setup_logging(config, game.params)

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

    def _setup_logging(self, pop_params, game_params):
        self.multilogger = MultiLogger(folder=self._folder)

        info = {"population_config": pop_params}
        info.update({"metagame_info": game_params})

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
        self.epoch += 1

        try:
            n_agents = next(self.n_agents)
        except StopIteration:
            self.terminated = True
            return

        prev_gen = [(self.population[i], f[i]) for i in range(len(f))]
        prev_gen = sorted(prev_gen, key=lambda x: x[1])
        prev_gen = [value[0] for value in prev_gen if value[0]["_age"] < self._max_age]

        self.population = []

        self.population += [
            self._random() for _ in range(int(n_agents * self._random_rate))
        ]

        if int(n_agents * self._survivor_rate):  # -0 returns whole list!!
            survivors = [
                deepcopy(genotype)
                for genotype in prev_gen[-int(n_agents * self._survivor_rate) :]
            ]

            for genotype in survivors:
                genotype["_age"] += 1

            self.population += survivors

        mutate_candidates = prev_gen[-int(self._mutate_eligable_pct * len(prev_gen)) :]
        self.population += self._mutate(
            [
                deepcopy(genotype)
                for genotype in np.random.choice(
                    mutate_candidates, size=int(n_agents * self._mutation_rate)
                )
            ]
        )

        for _ in range(int(n_agents * self._crossover_rate) // 2):
            genotype1 = np.random.choice(prev_gen)
            genotype2 = np.random.choice(prev_gen)

            self.population += self._crossover(deepcopy(genotype1), deepcopy(genotype2))

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
        params = [
            (
                self.get_fitness,
                self.cache,
                genotype,
                self._log_fn,
                next(self.multilogger.filename_generator) if self._logging else None
            )
            for genotype in self.population
        ]

        results = self.backend.distribute(run, params)

        fitnesses = [result[0] for result in results]
        terminated = [result[1] for result in results]

        if any(terminated):
            self.terminated = True

        return fitnesses
