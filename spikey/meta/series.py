"""
Tools for running series of expeirments.

Configuration
-------------
Any single type of confirugation can be given at at time to
experiment_params parameter.

- (attr, startval=0, stopval, step=1) -> np.arange(\*v[1:])
- (attr, [val1, val2...]) -> (i for i in val)
- (attr, generator) -> (i for i in val)
- (attr, func/iterable obj) -> (i for i in val)
- [(), ()] -> Multiple params iterated over together.

Examples
--------

.. code-block:: python

    experiment_list = {
        "control": None,
        'trace_decay': ('trace_decay', [.01, .02, .03, .04, .05]),
    }

    for experiment_name, series_params in experiment_list.items():
        folder = os.path.join("log", f"{experiment_name}")

        series = spikey.meta.Series(
            spikey.core.TrainingLoop(network_template, game_template, training_params),
            series_params,
            max_process=2,
        )

        series.run(n_repeats=2, log_folder=folder)
"""
from copy import deepcopy
import numpy as np
from spikey.module import Module
from spikey.logging import log, MultiLogger
from spikey.meta.backends.default import MultiprocessBackend


def run(training_loop: object, filename: str, log_fn: callable = log):
    output = training_loop()
    if filename is not None:
        log_fn(*output, filename=filename)
    return output


class Series(Module):
    """
    An experiment generator.

    Parameters
    ----------
    training_loop: TrainingLoop
        Configured training loop used in experiments.
    experiment_params: tuple or list
        Experiment parameter generators, see below.
    backend: MetaBackend, default=MultiprocessBackend(max_process)
        Backend to execute experiments with.
    max_process: int, default=16
        Number of separate processes to run experiments for
        default backend.
    logging: bool, default=True
        Whether to log results or not.

    Configuration
    -------------
    Any single type of confirugation can be given at at time to
    experiment_params parameter.

    - (attr, startval=0, stopval, step=1) -> np.arange(\*v[1:])
    - (attr, [val1, val2...]) -> (i for i in val)
    - (attr, generator) -> (i for i in val)
    - (attr, func/iterable obj) -> (i for i in val)
    - [(), ()] -> Multiple params iterated over together.

    Examples
    --------

    .. code-block:: python

        experiment_list = {
            "control": None,
            'trace_decay': ('trace_decay', [.01, .02, .03, .04, .05]),
        }

        for experiment_name, series_params in experiment_list.items():
            folder = os.path.join("log", f"{experiment_name}")

            series = spikey.meta.Series(
                spikey.core.TrainingLoop(network_template, game_template, training_params),
                series_params,
                max_process=2,
            )

            series.run(n_repeats=2, log_folder=folder)
    """

    def __init__(
        self,
        training_loop: object,
        experiment_params: dict,
        backend: object = None,
        max_process: int = 16,
        logging: bool = True,
    ):
        super().__init__(**{})

        self.training_loop = training_loop
        self.backend = backend or MultiprocessBackend(max_process)
        self.experiment_params = experiment_params
        self.logging = logging

        if experiment_params is None:
            self.attrs = None
            self.param_gen = [None for _ in range(1)]
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
        self.param_gen = list(zip(*iterables))

    def __iter__(self) -> object:
        """
        Iterate over experiments in series.

        Yields
        -------
        Experiment
        """
        self.training_loop.reset()
        for values in self.param_gen:
            training_loop = self.training_loop.copy()

            if isinstance(self.attrs, tuple):
                training_loop.reset(**dict(zip(self.attrs, values)))
            elif self.attrs:
                training_loop.reset(**{self.attrs: values})

            yield training_loop

    def run(self, n_repeats: int, log_folder: str = "log") -> list:
        """
        Run all experiments in the series for n_repeats times.

        Parameters
        ----------
        n_repeats: int
            Number of times to repeat each experiment.
        log_folder: str, default="log"
            Folder to save logs.

        Returns
        -------
        list List of individual training loop results.
        """
        if self.logging:
            L = MultiLogger(folder=log_folder)

        params = [
            (experiment, next(L.filename_generator) if self.logging else None)
            for experiment in self
            for _ in range(n_repeats)
        ]

        results = self.backend.distribute(run, params)

        if self.logging:
            L.summarize({"experiment_params": self.experiment_params})

        return results
