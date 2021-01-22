"""
Meta N Queens, benchmark average episodes to solution.
"""
import time
import numpy as np

from spikey.meta import Population
from spikey.MetaRL import *


if __name__ == "__main__":
    game = MetaNQueens(8)

    pop_config = {
        "n_process": 1,
        "n_storing": 200,
        "n_agents": 100,
        "n_epoch": 3000,
        "mutate_eligable_pct": 0.5,
        "max_age": 4,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
    }
    N_SIMULATION = 1000
    final_epochs = []

    for _ in range(N_SIMULATION):

        population = Population(*game.population_arguments, **pop_config)

        # start = time()
        epoch = 0

        while not population.terminated:
            epoch += 1

            fitness = population.evaluate()

            population.update(fitness)

        final_epochs.append(epoch)

        print(f"{epoch} - Max fitness: {max(fitness)}")

        # print(time() - start, "seconds")

    mean_epoch = sum(final_epochs) // N_SIMULATION

    print(f"Mean epoch to success: {mean_epoch}")

    print(f"Successfull {np.sum(np.array(final_epochs) < 3000 - 1)}")
