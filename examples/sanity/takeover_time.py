"""
Evolving a neural network to be able to learn an RL enviornment.
"""
import numpy as np

from spikey.meta import Population
from spikey.core import GenericLoop
from spikey.MetaRL import *


if __name__ == "__main__":
    GENOTYPE_CONSTRAINTS = {
        "potential_decay": (0, 1),
        "prob_rand_fire": (0, 0.2),
        "learning_rate": (0, 1),
        "trace_decay": (0, 1),
        "firing_threshold": (0.5, 3),
        "processing_time": list(range(6, 20 + 1)),
        "input_firing_steps": list(range(5, 20 + 1)),
        "refractory_period": list(range(1, 5 + 1)),
        "stdp_window": list(range(1, 10 + 1)),
        "punish_mult": list(range(-1, 0 + 1)),
    }

    BASE_GENOTYPE = {
        "potential_decay": 0.9,
        "prob_rand_fire": 0.2,
        "learning_rate": 0.9,
        "trace_decay": 0.9,
        "firing_threshold": 2.7,
        "processing_time": 6,
        "input_firing_steps": 6,
        "refractory_period": 5,
        "stdp_window": 2,
        "punish_mult": 0,
    }

    pop_config = {
        "n_process": 1,
        "n_storing": 512,
        "n_agents": 256,
        "n_epoch": 250,
        "mutate_eligable_pct": 0.5,
        "max_age": 5,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
    }

    metagame = EvolveNetwork(
        network_template=None,
        game_template=None,
        win_fitness=100,
        training_loop=GenericLoop,
        fitness_getter=None,
        n_episodes=None,
        len_episode=None,
        n_reruns=None,
        genotype_constraints=GENOTYPE_CONSTRAINTS,
        static_config=None,
    )
    population = Population(*metagame.population_arguments, **pop_config, logging=False)

    TAKEOVER_PCT = 0.75

    RWD_DIST = 5

    def get_fitness(genotype, log=None, filename=None, q=None, reduced_logging=True):
        return int(population._genotype_dist(BASE_GENOTYPE, genotype) < RWD_DIST), False

    metagame.get_fitness = get_fitness
    population.get_fitness = get_fitness

    while not population.terminated:
        fitness = population.evaluate()

        print(f"{population.epoch}: {np.mean(fitness):.3f}")

        if np.mean(fitness) >= TAKEOVER_PCT:
            print(f"Takeover took {population.epoch} epochs!")
            break

        population.update(fitness)
    else:
        print(f"Failed to takeover after {population.epoch} epochs!")
