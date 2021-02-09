"""
Run a set of experiments -- optionally in parallel.
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
        #'stdp_window': ('stdp_window', [20, 30, 40, 50]),
        #'learning_rate': ('learning_rate', [.025, .075, .15, .25]),
    }
    N_REPEATS = 2

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
