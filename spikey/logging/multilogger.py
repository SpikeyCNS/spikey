"""
Logging multiple games over time.
"""
from datetime import datetime, timedelta
import os
from string import ascii_lowercase
import time

from spikey.logging.log import log


class MultiLogger:
    """
    A context manager to handle the logging of one experiment over time.

    Parameters
    ----------
    folder: str, default='log'
        Change folder to save in.
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
        Filename generator.

        Max 18k filenames.
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
        Save network and game data.

        Parameters
        ----------
        results: dict, default=None
            Results to log.
        info: dict, default=None
            Info to log.

        Post
        ----
        Saves .json in folder
        """
        filename = f"{self.prefix}~{filename_extension}.json"

        log(None, None, results, info, filename=filename)

    def log(self, network: "SNN", game: "RL", results: dict = None, info: dict = None):
        """
        Save network and game data.

        Parameters
        ----------
        network: SNN
            Network of interest.
        game: Game
            Played game.
        results: dict, default=None
            Results to log.
        info: dict, default=None
            Info to log.

        Post
        ----
        Saves .json in folder
        """
        log(network, game, results, info, filename=next(self.filename_generator))
