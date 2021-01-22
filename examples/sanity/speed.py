"""
Speed test -- do not change network settings!
"""
import os
from time import time

from spikey.experiments.benchmark import (
    network_template,
    game_template,
    training_params,
    Loop,
)


if __name__ == "__main__":
    N_REPEATS = 3

    experiment = Loop(network_template, game_template, training_params)

    start_time = time()

    for _ in range(N_REPEATS):
        network, game, results, info = experiment()

    total_time = time() - start_time
    average = total_time / N_REPEATS

    tick_time = average / (
        training_params["n_episodes"] * training_params["len_episode"]
    )
    stdp_time = tick_time / network.config["processing_time"]

    ## Output
    print(
        f"Simulation Run Time: {average}, Update Time {tick_time}, Step Time: {stdp_time}"
    )

    ## Save data
    from logger.log import log

    results.update(
        {
            "N_EPISODES": training_params["n_episodes"],
            "LEN_EPISODE": training_params["len_episode"],
            "N_REPEATS": N_REPEATS,
            "Mean Simulation Time": average,
            "Time per Update": tick_time,
            "Step Time": stdp_time,
        }
    )
    log(network, game, results, info, folder=os.path.join("examples", "sanity"))
