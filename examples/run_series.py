"""
Run a set of experiments -- optionally in parallel.

NOTE: Windows may require to run in administrator.
"""
import os
import spikey
from spikey.experiments.florian_rate import (
    network_template,
    game_template,
    training_params,
)


if __name__ == "__main__":
    experiment_list = {
        "control": None,
        #'trace_decay': ('trace_decay', [.01, .02, .03, .04, .05]),
        #'potential_decay': ('potential_decay', [.05, .1, .2, .4, .6, .8]),
        #'prob_rand_fire': ('prob_rand_fire', [0, .05, .1, .25]),
        #'refractory_period': ('refractory_period', [0, 1, 3, 6]),
        #'stdp_window': ('stdp_window', [20, 30, 40, 50]),
        #'learning_rate': ('learning_rate', [.025, .075, .15, .25]),
        #'input': [('input', [UniformSpikeGenerator, PoissonSpikeGenerator]), ('rate_mapping', [[0, .4], [0, .165]])],
    }
    N_REPEATS = 4

    MAX_PROCESS = 2

    for name, series_params in experiment_list.items():
        series = spikey.meta.Series(
            spikey.core.GenericLoop,
            network_template,
            game_template,
            training_params,
            series_params,
            max_process=MAX_PROCESS,
        )

        folder = os.path.join("log", f"{name}")

        series.run(N_REPEATS, folder)
