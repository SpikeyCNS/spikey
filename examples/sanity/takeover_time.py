"""
Measure the time it takes for a high fitness genotype to dominate
the gene pool. Ideally it will quickly take over a large portion
of the next generations but not entirely dominate for a long time.
"""
import numpy as np

from spikey.meta import Population
from spikey.core import TrainingLoop
from spikey.MetaRL import EvolveNetwork


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
        "max_process": 1,
        "n_storing": 512,
        "n_agents": 256,
        "n_epoch": 250,
        "mutate_eligable_pct": 0.5,
        "max_age": 5,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "logging": False,
    }

    game = EvolveNetwork(
        training_loop=TrainingLoop(None, None),
        genotype_constraints=GENOTYPE_CONSTRAINTS,
        win_fitness=100,
        fitness_getter=None,
    )
    population = Population(game=game, **pop_config)

    TAKEOVER_PCT = 0.75

    RWD_DIST = 5

    def get_fitness(genotype, log=None, filename=None, q=None, reduced_logging=True):
        return int(population._genotype_dist(BASE_GENOTYPE, genotype) < RWD_DIST), False

    game.get_fitness = get_fitness
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
