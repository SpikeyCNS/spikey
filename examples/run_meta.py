"""
Evolving a neural network to be able to learn an RL enviornment.
"""
import time
import numpy as np
import os

from spikey.core import GenericLoop
from spikey.meta import Population, checkpoint_population
from spikey.MetaRL import *


## NOTE: Functions that are to be multiprocessed need to
#       be top level -- ie here. See what can be pickled.
def fitness_getter(network, game, results, info):
    return results["total_time"]


if __name__ == "__main__":
    from spikey.experiments.florian_rate import (
        network_template as network,
        game_template as game,
        training_params,
    )

    network.config.update({"processing_time": 50})

    STATIC_UPDATES = ("prob_rand_fire", [0.0, 0.0, 0.02, 0.02, 0.04])

    GENOTYPE_CONSTRAINTS = {
        "input_pct_inhibitory": list(np.arange(0, 1, 0.05)),
        "neuron_pct_inhibitory": list(np.arange(0, 1.0, 0.05)),
        "firing_threshold": list(range(1, 31)),
        "potential_decay": list(np.arange(0, 1, 0.02)),
        "trace_decay": list(np.arange(0, 1, 0.02)),
        "refractory_period": list(range(15)),
        "max_weight": list(np.arange(1, 10.1, 0.5)),
        "stdp_window": list(range(5, 100, 5)),
        "learning_rate": [x / 25 for x in np.arange(0.01, 1.0, 0.01)],
        "magnitude": list(np.arange(-10, 10.1, 0.5)),
        "spike_delay": list(range(10)),
        "reward_mult": list(np.arange(0, 5.1, 0.5)),
        "punish_mult": list(np.arange(0, 5.1, 0.5)),
    }

    metagame_config = {
        "win_fitness": 0.9,
        "fitness_getter": fitness_getter,
        "n_reruns": 5,
        "genotype_constraints": GENOTYPE_CONSTRAINTS,
        "static_updates": STATIC_UPDATES,
    }

    pop_config = {
        "folder": os.path.join("log", "metarl"),
        "max_process": 2,
        "n_storing": 256,
        "n_agents": 4,
        "n_epoch": 1,
        "mutate_eligable_pct": 0.5,
        "max_age": 5,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
    }
    metagame = EvolveNetwork(
        training_loop=GenericLoop(network, game, **training_params),
        **metagame_config,
    )
    population = Population(
        game=metagame,
        **pop_config,
    )

    # spikey.meta.read_population(population, os.path.join('log', 'metarl'))

    start = time.time()
    while not population.terminated:
        fitness = population.evaluate()
        population.update(fitness)

        checkpoint_population(population)

        print(f"{population.epoch} - Max fitness: {max(fitness)}")

    print(f"{time.time() - start:.2f}s")
