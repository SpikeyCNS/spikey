"""
Evolving a neural network to be able to learn an RL enviornment.
"""
import time
import numpy as np
import os

from spikey.core import GenericLoop
from spikey.meta import Population
from spikey.MetaRL import *


N_INPUTS = 60
N_NEURONS = 61
N_OUTPUTS = 1

## NOTE: Functions that are to be multiprocessed need to
#       be top level -- ie here. See what can be pickled.
def make_w():
    w = np.vstack(
        (  # Fully connected, generated randomly over interval
            np.hstack(
                (
                    np.random.uniform(0, 5, (N_INPUTS, N_NEURONS - N_OUTPUTS)),
                    np.zeros((N_INPUTS, N_OUTPUTS)),
                )
            ),
            np.hstack(
                (
                    np.zeros((N_NEURONS - N_OUTPUTS, N_NEURONS - N_OUTPUTS)),
                    np.random.uniform(0, 5, (N_NEURONS - N_OUTPUTS, N_OUTPUTS)),
                )
            ),
            np.zeros((N_OUTPUTS, N_NEURONS)),
        )
    )
    w = np.ma.array(np.float16(w), mask=(w == 0))

    return w


def tracking_getter(n, g, r, i):
    return r["total_time"]


def expected_value(state):
    return np.sum(state) % 2


if __name__ == "__main__":
    from spikey.experiments.florian_rate import (
        network_template as network,
        game_template as game,
        training_params,
    )

    matricies = [make_w() for i in range(5)]

    STATIC_UPDATES = ("matrix", matricies)

    # STATIC_FILENAME = ".static_updates.obj"

    ## Preserve static_updates
    # from pickle import dump
    # with open(STATIC_FILENAME, "wb") as file:
    #    dump(STATIC_UPDATES, file)

    ## Load static_updates
    # from pickle import load
    # with open(STATIC_FILENAME, 'rb') as file:
    #   STATIC_UPDATES = load(file)

    network.config = {}
    game._params = {}

    STATIC_CONFIG = {
        "matrix": None,
        "inh_weight_mask": None,
        "n_neurons": N_NEURONS,
        "processing_time": 50,
        "resting_mv": 0.0,
        "output_range": [0, 1],
        "action_threshold": 0,
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "firing_steps": -1,
        "rate_mapping": [0, 0.08],
        "prob_rand_fire": 0,
        "reward_mult": 1,
        "punish_mult": 1,
        "expected_value": expected_value,
    }

    ## NOTE: +1 for all randint
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
        "florian_reward": list(np.arange(0, 5.1, 0.5)),
        "florian_punish": list(np.arange(0, -5.1, -0.5)),
    }

    metagame_config = {
        "win_fitness": 0.9,
        "tracking_getter": tracking_getter,
        "n_reruns": 5,
        "genotype_constraints": GENOTYPE_CONSTRAINTS,
        "static_config": STATIC_CONFIG,
        "static_updates": STATIC_UPDATES,
    }
    metagame_config.update(training_params)

    pop_config = {
        "n_process": 2,
        "folder": os.path.join("log", "metarl"),
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
        network_template=network,
        game_template=game,
        training_loop=GenericLoop,
        **metagame_config,
    )
    population = Population(
        *metagame.population_arguments,
        **pop_config,
        log_info=metagame_config,
        logging=True,
        reduced_logging=True,
    )

    # population.read_population(os.path.join('log', 'metarl'))

    try:
        start = time.time()

        while not population.terminated:
            fitness = population.evaluate()

            population.update(fitness)

            print(f"{population.epoch} - Max fitness: {max(fitness)}")

        print(time.time() - start)
    except KeyboardInterrupt:
        pass
