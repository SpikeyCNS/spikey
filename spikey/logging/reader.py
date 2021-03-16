"""
Read a single or set of experiment log files.

Reader - Read group of files.
MetaReader - Read group of files in MetaRL format.
"""
import os
import json
import pandas as pd
import numpy as np
from spikey.logging.serialize import uncompressnd


def dejsonize(value: str) -> object:
    """
    Convert value from string as read from json into type similar to original.

    Parameters
    ----------
    value: str
        Value to convert.

    Returns
    -------
    any Value converted into suspected original type.
    """
    if isinstance(value, type(None)):
        return value
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = dejsonize(v)
        return value
    elif isinstance(value, (list, tuple)):
        return [dejsonize(v) for v in value]

    try:
        return float(value)
    except:
        pass

    if "[" in value and "]" in value and " " in value:
        return uncompressnd(value)

    return value


class Reader:
    """
    Read group of files or a single file.

    Parameters
    ----------
    folder: str, default="."
        Folder to read from.
    filenames: list, default=os.listdir(foldre)
        List of specific filenames in the folder to read.

    Examples
    --------

    .. code-block:: python

        reader = Reader('log')

        df = reader.df
        states = reader['step_states']

    .. code-block:: python

        reader = Reader('example.json')

        df = reader.df
        states = reader['step_states']
    """

    COLUMNS = ["snn", "game", "results"]

    def __init__(self, folder: str, filenames: list = None):
        if folder.endswith(".json"):
            self.folder = "."
            self.filenames = [folder]
        else:
            self.folder = folder
            self.filenames = (
                filenames if filenames is not None else os.listdir(self.folder)
            )

        ## __a before aaa (os.listdir reads in opposite order)
        self.filenames.sort()

        self.output = None

        for i, filename in enumerate(self.filenames):
            with open(os.path.join(self.folder, filename), "r") as file:
                try:
                    data = json.load(file)
                except UnicodeDecodeError:
                    print(f"Failed to read '{os.path.join(self.folder, filename)}'!")
                    continue

                if "SUMMARY" in filename:
                    self.summary = {
                        key: dejsonize(value) for key, value in data.items()
                    }

                else:
                    store = {}
                    for column in self.COLUMNS:
                        if column not in data:
                            continue

                        store.update(
                            {
                                key: dejsonize(value)
                                for key, value in data[column].items()
                            }
                        )

                    if self.output is None:
                        self.output = pd.DataFrame(columns=list(store))
                        self._index_keys(data)

                    if len(store):
                        self.output.loc[i] = [
                            store[key] if key in store else np.nan
                            for key in self.output.columns
                        ]

    @property
    def network(self) -> dict:
        """
        Return network parameters from experiment.
        """
        return self.df[self._snn_keys]

    @property
    def game(self) -> dict:
        """
        Return game parameters from experiment.
        """
        return self.df[self._game_keys]

    @property
    def results(self) -> dict:
        """
        Return results from experiment.
        """
        return self.df[self._results_keys]

    @property
    def df(self) -> pd.DataFrame:
        return self.output

    def __len__(self) -> int:
        return len(self.df)

    def _index_keys(self, data):
        for column in self.COLUMNS:
            if column not in data:
                continue
            keys = data[column].keys()
            setattr(self, f"_{column}_keys", keys)

    def iter_unique(
        self, key: str, hashable_keys: str = None, return_value: bool = False
    ) -> pd.DataFrame:
        """
        Iterator through unique values of key.

        Parameters
        ----------
        key: str/list
            Label(s) to list unique configurations to iterate over.
        hashable_keys: str, default=None
            Keys eligable as selector & return values.
        return_value: bool, default=False
            Whether to yield (unique_value, params) or just params

        Yields
        ------
        (Series, DataFrame) or DataFrame for a unique set in config[key].
        """
        if hashable_keys is None:
            hashable_keys = key

        unique_params = self.df[hashable_keys].drop_duplicates(inplace=False)

        for value in unique_params.iterrows():
            if return_value:
                yield value, self[
                    key,
                    np.where((self.df[hashable_keys] == value[1]).all(axis="columns"))[
                        0
                    ],
                ]
            else:
                yield self[
                    key,
                    np.where((self.df[hashable_keys] == value[1]).all(axis="columns"))[
                        0
                    ],
                ]

    def read_info(self, attr: str, n: int, loc: int = 0) -> list:
        """
        Pull values for given key from file.

        Parameters
        -----------
        attr: str
            Key for values desired.
        n: int
            Number of files to read, from loc to loc+n.
        loc: int or iterable, default=0
            Offset for starting filename or locations of filenames desired.

        Returns
        -------
        list Each files value for given attribute.

        Examples
        --------

        .. code-block:: python

            reader = Reader('log')

            states = reader.read_info('step_states', len(reader))
        """
        try:
            iter(n)
            relevant_filenames = [self.filenames[nn] for nn in n]
        except TypeError:
            relevant_filenames = self.filenames[loc : loc + n]

        out = []

        for i, filename in enumerate(relevant_filenames):
            if "SUMMARY" in filename:
                continue

            with open(os.path.join(self.folder, filename), "r") as file:
                data = json.load(file)

                out.append(dejsonize(data["info"][attr]))

        return out

    def read_file(self, filename: str) -> dict:
        """
        Read all data in specific file.

        Parameters
        -----------
        filename: str
            Filename to read.

        Returns
        -------
        dict All data from file.

        Examples
        --------

        .. code-block:: python

            data = Reader('log').read_file('experiment_log.json')
        """
        with open(os.path.join(self.folder, filename), "r") as file:
            try:
                data = json.load(file)
            except UnicodeDecodeError:
                print(f"Failed to read '{os.path.join(self.folder, filename)}'!")
                return

            store = {}
            for column in data:
                try:
                    store.update({key: value for key, value in data[column].items()})
                except AttributeError:
                    store.update({column: data[column]})

        return store

    def __getitem__(self, key: tuple) -> object:
        """
        Pull values for given key from dataframe or file as necessary.

        Parameters
        -----------
        key: str or (key: str, n_entries: int)
            Key for values desired.

        Returns
        -------
        dict or dataframe Gathered values from key given. Type depends
        respectively on whether key is in info section, which requires
        pulling from file or, in network, game or results table.

        Examples
        --------

        .. code-block:: python

            reader = Reader('log')

            n_neurons = reader['n_neurons']
            states = reader['step_states']
        """
        if not isinstance(key, tuple):
            key = (key, len(self.output))
        elif len(key) < 2:
            key = (*key, len(self.output))

        param, v = key

        try:
            if isinstance(param, str) and param not in self.df:
                raise TypeError()

            iter(param)
        except TypeError:
            return self.read_info(param, v)

        try:
            iter(v)

            return self.df[param].loc[v]
        except TypeError:
            return self.df[param][:v]


class MetaReader(Reader):
    """
    Read group of files or a single file in the MetaRL format.

    Parameters
    ----------
    folder: str, default="."
        Folder to read from.
    filenames: list, default=os.listdir(foldre)
        List of specific filenames in the folder to read.

    Examples
    --------

    .. code-block:: python

        reader = MetaReader('log')

        print(reader.summary)  # -> Summary file contents

        df = reader.df
        max_fitness = np.max(df['fitness'])
        print(max_fitness)

        max_params = df[df['fitness'] == max_fitness]
        for param in max_params:
            print(param)
    """

    COLUMNS = ["snn", "game", "results", "info"]

    def __init__(self, folder: str = ".", filenames: list = None):
        self.folder = folder
        self.filenames = filenames if filenames is not None else os.listdir(self.folder)

        self.filenames.sort()

        self.output = None

        for i, filename in enumerate(self.filenames):
            if "SUMMARY" in filename:
                with open(os.path.join(self.folder, filename), "r") as file:
                    data = json.load(file)

                self.summary = {key: dejsonize(value) for key, value in data.items()}

                self.filenames.remove(filename)

        self.genotype_keys = list(
            self.summary["info"]["metagame_info"]["genotype_constraints"].keys()
        )

        for i, filename in enumerate(self.filenames):
            if "SUMMARY" in filename:
                continue

            with open(os.path.join(self.folder, filename), "r") as file:
                try:
                    data = json.load(file)
                except UnicodeDecodeError:
                    print(f"Failed to read '{os.path.join(self.folder, filename)}'!")
                    continue

                store = {}
                store.update({"filename": filename})

                for column1 in self.COLUMNS:
                    if column1 not in data:
                        continue
                    iterable = (
                        self.genotype_keys
                        if column1 == "info"
                        else data[column1].keys()
                    )
                    for column2 in iterable:
                        try:
                            store.update({column2: dejsonize(data[column1][column2])})
                            continue
                        except KeyError:
                            pass

                store.update({"fitness": dejsonize(data["results"]["fitness"])})

                if self.output is None:
                    self.output = pd.DataFrame(columns=list(store.keys()))
                    self._index_keys(data)

                self.output.loc[i] = [store[key] for key in self.output.columns]
