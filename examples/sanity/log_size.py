"""
Tool to see how big generated log files are.
"""
import os

from spikey.experiments.benchmark import (
    network_template,
    game_template,
    training_params,
    Loop,
)

from spikey.logging import log


FOLDER = os.path.abspath(__file__).split("sanity")[0]


def measure_log(network, game, results=None, info=None):
    filename = log(network, game, results, info, folder=FOLDER)

    n_bytes = os.stat(filename).st_size

    os.remove(filename)

    return n_bytes


if __name__ == "__main__":
    experiment = Loop(network_template, game_template, training_params)

    network, game, results, info = experiment()
    print(
        f"Standard log is {measure_log(network, game, results, info) / 1000:.1f} Kbytes"
    )

    network, game, results, info = experiment(reduced=True)
    print(
        f"Reduced log is {measure_log(network, game, results, info) / 1000:.1f} Kbytes"
    )
