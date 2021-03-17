"""
Tool to manage logging for series of experiments.
"""
from datetime import datetime, timedelta
import os
from string import ascii_lowercase
import time

from spikey.logging.log import log


class MultiLogger:
    """
    Context manager for logging a series of experiments.

    Parameters
    ----------
    folder: str, default="log"
        Folder to save logs into.

    Examples
    --------

    .. code-block:: python

        experiment = TrainingLoop(Network, RL, **config)
        logger = MultiLogger()

        for _ in range(10):
            network, game, results, info = experiment()
            logger.log(network, game, results, info)

        logger.summary()

    .. code-block:: python

        callback = ExperimentCallback()
        experiment = TrainingLoop(Network, RL, callback, **config)

        with MultiLogger(folder="log") as logger:
            for _ in range(10)
                experiment()
                logger.log(*callback)

            logger.summary()
    """

    def __init__(self, folder: str = None):
        ## Filename generator
        self.folder = folder or "log"

        try:
            os.makedirs(self.folder)

            print(f"Created directory {self.folder}!")
        except FileExistsError:
            pass

        date = datetime.now()
        while any(
            [
                date.strftime("%Y-%m-%d-%H-%M") in filename
                for filename in os.listdir(self.folder)
            ]
        ):
            date += timedelta(minutes=1)

        self.prefix = os.path.join(self.folder, f"{date.strftime('%Y-%m-%d-%H-%M')}")

        self.filename_generator = self.filename_gen()

        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

        return self

    def __exit__(self, *_):
        print(f"Process took {time.time() - self.start_time} seconds.")

    def filename_gen(self) -> str:
        """
        Generate filenames in structure f"{prefix}-{UUID}.json".
            UUID: Ordered 5 letter strings from "_____" to "zzzzz", up to 18k.
            Prefix: join(self.folder, {YY}-{MM}-{DD}-{HH}-{MM})

        Examples
        --------

        .. code-block:: python

            experiment = TrainingLoop()
            filename_generator = Multilogger().filename_gen()

            for _ in range(10):
                network, game, results, info = experiment()
                log(network, game, results, info, filename=next(filename_generator))
        """
        for letter1 in ["_"] + list(ascii_lowercase):
            for letter2 in ["_"] + list(ascii_lowercase):
                for letter3 in ["_"] + list(ascii_lowercase):
                    for letter4 in ["_"] + list(ascii_lowercase):
                        for letter5 in ascii_lowercase:
                            yield f"{self.prefix}-{letter1}{letter2}{letter3}{letter4}{letter5}.json"

    def summarize(
        self,
        results: dict = None,
        info: dict = None,
        filename_extension: str = "SUMMARY",
    ):
        """
        Log summary of experiment series to file.

        Parameters
        ----------
        results: dict, default={}
            Set of dataframe compatible results.
        info: dict, default={}
            Data meant for further analysis.
        filename_extension: str, default="SUMMARY"
            Identifier added to multilogger prefix to get summary filename.

        Examples
        --------

        .. code-block:: python

            experiment = TrainingLoop(Network, RL, **config)
            logger = MultiLogger()

            for _ in range(10):
                network, game, results, info = experiment()
                logger.log(network, game, results, info)

            logger.summary()

        .. code-block:: python

            callback = ExperimentCallback()
            experiment = TrainingLoop(Network, RL, callback, **config)

            with MultiLogger(folder="log") as logger:
                for _ in range(10)
                    experiment()
                    logger.log(*callback)

                logger.summary()
        """
        filename = f"{self.prefix}~{filename_extension}.json"

        log(None, None, results, info, filename=filename)

    def log(
        self,
        network: object,
        game: object,
        results: dict = None,
        info: dict = None,
        log_fn: callable = log,
    ) -> str:
        """
        Log experiment data to file.

        .. code-block:: python

            {
                'metadata': value,
                'snn': {
                    Network configuration data.
                },
                'game': {
                    Game configuration data.
                },
                'results': {
                    Results, values that can be directly loaded to table.
                },
                'info': {
                    Data meant for further analysis.
                    Not loaded in table by Reader.
                }
            }

        Parameters
        ----------
        network: SNN
            Network used in experiment.
        game: Game
            Game played in experiment.
        results: dict, default={}
            Set of dataframe compatible results.
        info: dict, default={}
            Data meant for further analysis.
        log_fn: callable, default=log
            Function used to log data.

        Returns
        -------
        str Filename logged to.

        Examples
        --------

        .. code-block:: python

            experiment = TrainingLoop(Network, RL, **config)
            logger = MultiLogger()

            for _ in range(10):
                network, game, results, info = experiment()
                logger.log(network, game, results, info)

            logger.summary()

        .. code-block:: python

            callback = ExperimentCallback()
            experiment = TrainingLoop(Network, RL, callback, **config)

            with MultiLogger(folder="log") as logger:
                for _ in range(10)
                    experiment()
                    logger.log(*callback)

                logger.summary()
        """
        return log_fn(
            network, game, results, info, filename=next(self.filename_generator)
        )
