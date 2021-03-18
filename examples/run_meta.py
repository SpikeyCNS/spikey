"""
Evolve a neural network to learn an RL enviornment.
"""
import time
import numpy as np
import os

from spikey.core import GenericLoop
from spikey.meta import Population, checkpoint_population
from spikey.MetaRL import EvolveNetwork


## NOTE: Functions to be multiprocessed need be top level -- eg here.
def fitness_getter(network, game, results, info):
    return results["total_time"]  # TODO Replace this with your fitness


if __name__ == "__main__":
    from spikey.experiments.florian_rate import (
        network_template as network,
        game_template as game,
        training_params,
    )

    network.keys.update({"processing_time": 50})

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
        "reward_mult": list(np.arange(0, 5.1, 0.5)),
        "punish_mult": list(np.arange(0, 5.1, 0.5)),
    }

    metagame_config = {
        "win_fitness": 9999,
        "fitness_getter": fitness_getter,
        "n_reruns": 5,  # 1 run per static update
        "static_updates": ("prob_rand_fire", [0.0, 0.0, 0.02, 0.02, 0.04]),
        "genotype_constraints": GENOTYPE_CONSTRAINTS,
    }

    metagame = EvolveNetwork(
        training_loop=GenericLoop(network, game, **training_params),
        **metagame_config,
    )

    pop_config = {
        "folder": os.path.join("log", "metarl"),
        "n_epoch": 1,
        "n_agents": 4,
        "max_process": 2,
        "n_storing": 256,
        "mutate_eligable_pct": 0.5,
        "max_age": 5,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
    }
    population = Population(
        game=metagame,
        **pop_config,
    )

    # population = spikey.meta.read_population()

    start = time.time()
    while not population.terminated:
        fitness = population.evaluate()
        population.update(fitness)

        checkpoint_population(population)

        print(f"{population.epoch} - Max fitness: {max(fitness)}")

    print(f"{time.time() - start:.2f}s")
