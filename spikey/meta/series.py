"""
Tools for running series of expeirments.

Configuration
-------------
Any single type of confirugation can be given at at time to
experiment_params parameter.

* (attr, startval=0, stopval, step=1) -> np.arange(*v[1:])
* (attr, [val1, val2...]) -> (i for i in val)
* (attr, generator) -> (i for i in val)
* (attr, func/iterable obj) -> (i for i in val)
* [(), ()] -> Multiple params iterated over together.

Usage
-----
experiment_list = {
    "control": None,
    'trace_decay': ('trace_decay', [.01, .02, .03, .04, .05]),
}

for experiment_name, series_params in experiment_list.items():
    folder = os.path.join("log", f"{experiment_name}")

    series = spikey.meta.Series(
        spikey.core.TrainingLoop,
        network_template,
        game_template,
        training_params,
        series_params,
        max_process=2,
    )

    series.run(n_repeats=2, log_folder=folder)
"""
from copy import deepcopy
import numpy as np

from spikey.logging import log, MultiLogger
from spikey.meta.backends.default import MultiprocessBackend


get_alive = np.vectorize(lambda v: v.is_alive())


def run(training_loop: object, filename: str):
    network, game, results, info = training_loop(filename=filename)


class Series:
    """
    An experiment generator.

    Parameters
    ----------
    ControlSNN: SNN[class]
        Control network.
    ControlGame: RL[class]
        Control game.
    control_config: dict
        Baseline config for network and game.
    experiment_params: tuple or list
        Experiment parameter generators, see below.

    Configuration
    -------------
    Any single type of confirugation can be given at at time to
    experiment_params parameter.

    * (attr, startval=0, stopval, step=1) -> np.arange(*v[1:])
    * (attr, [val1, val2...]) -> (i for i in val)
    * (attr, generator) -> (i for i in val)
    * (attr, func/iterable obj) -> (i for i in val)
    * [(), ()] -> Multiple params iterated over together.
    """

    def __init__(
        self,
        trainingloop: "TrainingLoop",
        ControlSNN: type,
        ControlGame: type,
        control_config: dict,
        experiment_params: dict,
        backend: object = None,
        max_process: int = 16,
    ):
        self.trainingloop = trainingloop
        self.ControlSNN = ControlSNN
        self.ControlGame = ControlGame
        self.control_config = control_config
        self.backend = backend or MultiprocessBackend(max_process)

        if experiment_params is None:
            self.attrs = None
            self.param_gen = (None for _ in range(1))
            return

        if isinstance(experiment_params, tuple):
            experiment_params = [experiment_params]

        self.attrs = []
        iterables = []

        for param in experiment_params:
            name, first, values = param[0], param[1], param[1:]

            self.attrs.append(name)

            if isinstance(first, (int, float)):
                iterables.append((i for i in np.arange(*values)))
            elif isinstance(first, tuple):
                iterables.append((i for i in np.arange(*first)))
            elif isinstance(first, (list, np.ndarray)):
                iterables.append((i for i in first))
            elif callable(first) or hasattr(first, "__iter__"):
                iterables.append(first)
            else:
                raise ValueError("Failed to recognize type of iterable. {first}")

        self.attrs = tuple(self.attrs)
        self.param_gen = (i for i in zip(*iterables))

    def __iter__(self) -> object:
        """
        Yields
        -------
        Experiment
        """
        for values in self.param_gen:
            experiment_params = deepcopy(self.control_config)

            if isinstance(self.attrs, tuple):
                experiment_params.update(dict(zip(self.attrs, values)))
            elif self.attrs:
                experiment_params.update({self.attrs: values})

            yield self.trainingloop(
                self.ControlSNN, self.ControlGame, experiment_params
            )

    def run(self, n_repeats: int, log_folder: str = None):
        """
        Run all experiments in the series for n_repeats times.
        """
        log_folder = log_folder or "log"
        L = MultiLogger(folder=log_folder)

        params = [
            (experiment, next(L.filename_generator))
            for experiment in self
            for _ in range(n_repeats)
        ]

        results = self.backend.distribute(run, params)

        L.summarize()
