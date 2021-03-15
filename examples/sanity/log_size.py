"""
Tool to see how big generated log files are.
"""
import os

from spikey.core import GenericLoop, RLCallback
from spikey.experiments.benchmark import (
    network_template,
    game_template,
)
from spikey.logging import log


FOLDER = os.path.abspath(__file__).split("sanity")[0]


def measure_log(network, game, results=None, info=None):
    filename = log(network, game, results, info, folder=FOLDER)
    n_bytes = os.stat(filename).st_size
    os.remove(filename)
    return n_bytes


if __name__ == "__main__":
    training_params = {
        "n_episodes": 1,
        "len_episode": 100,
    }
    n_steps = training_params["n_episodes"] * training_params["len_episode"]
    training_loop = GenericLoop(network_template, game_template, **training_params)

    callback = RLCallback()
    training_loop.reset(callback=callback)
    network, game, results, info = training_loop()
    print(
        f"Standard log is {measure_log(network, game, results, info) / 1000:.1f}KB / {n_steps} steps"
    )

    callback = RLCallback(reduced=True)
    training_loop.reset(callback=callback)
    network, game, results, info = training_loop()
    print(
        f"Reduced log is {measure_log(network, game, results, info) / 1000:.1f}KB / {n_steps} steps"
    )
