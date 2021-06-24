"""
Meta N Queens benchmark counting average episodes to solution.
"""
import time
import numpy as np

from spikey.meta import Population
from spikey.meta.metagames import MetaNQueens


if __name__ == "__main__":
    game = MetaNQueens(n_queens=8)

    pop_config = {
        "max_process": 1,
        "n_storing": 200,
        "n_agents": 100,
        "n_epoch": 3000,
        "mutate_eligable_pct": 0.5,
        "max_age": 4,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
        "logging": False,
    }
    final_epochs = []
    for _ in range(100):
        population = Population(game=game, **pop_config)
        epoch = 0
        while not population.terminated:
            epoch += 1
            fitness = population.evaluate()
            population.update(fitness)
        final_epochs.append(epoch)
        print(f"{epoch} - Max fitness: {max(fitness)}")

    mean_epoch = sum(final_epochs) // N_SIMULATION
    print(f"Mean epoch to success: {mean_epoch}")
    print(f"Successfull {np.sum(np.array(final_epochs) < 3000 - 1)}")
