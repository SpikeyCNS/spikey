"""
Find max RAM usage of network.

-- NOTE: Requires pip install psutil
"""
import os
import psutil

from spikey.experiments.benchmark import (
    network_template,
    game_template,
    training_params,
    Loop,
)

usage = []


def memory_wrapper(func):
    def wrapped(*args, **kwargs):
        usage.append(py.memory_info()[0] / 1000 ** 2)

        return func(*args, **kwargs)

    return wrapped


if __name__ == "__main__":
    pid = os.getpid()
    py = psutil.Process(pid)

    network_template.tick = memory_wrapper(network_template.tick)

    experiment = Loop(network_template, game_template, training_params)

    experiment()

    print(f"Max ram usage: {max(usage):.1f} Mbytes")
