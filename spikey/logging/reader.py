"""
For reading groups of generated log files.

Reader - Read group of files.
MetaReader - Read group of files in MetaRL format.
"""
import os
import json
import pandas as pd
import numpy as np
from spikey.logging.serialize import uncompressnd


def dejsonize(value: str) -> dict:
    """
    Attempt to convert value from string to original/good enough type.
    """
    if isinstance(value, type(None)):
        return value

    if isinstance(value, dict):
        return value

    try:
        return float(value)
    except ValueError:
        pass

    if "[" in value and "]" in value and " " in value:
        return uncompressnd(value)

    return value


class Reader:
    """
    Read group of files or a single file.
    """

    COLUMNS = ["snn", "game", "results"]

    def __init__(self, folder: str, filenames: list = None):
        self.folder = folder
        self.filenames = filenames if filenames is not None else os.listdir(self.folder)

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

                    self.output.loc[i] = [
                        store[key] if key in store else np.nan
                        for key in self.output.columns
                    ]

    @property
    def df(self) -> pd.DataFrame:
        return self.output

    def iter_unique(
        self, key: str, hashable_keys: str = None, return_value: bool = False
    ) -> pd.Series:
        """
        Iterate through unique config values.

        key: str/list[str]
            Label you want to iterate through unique values for
        """
        if hashable_keys is None:
            hashable_keys = key

        unique_params = self.df[hashable_keys].drop_duplicates(inplace=False)

        for value in unique_params.iterrows():
            # yield self.df[(self.df[hashable_keys] == value[1]).all(axis='columns')][key]
            if return_value:
                yield value, self[
                    key,
                    np.where((self.df[hashable_keys] == value[1]).all(axis="columns"))[
                        0
                    ],
                ]
                continue

            yield self[
                key,
                np.where((self.df[hashable_keys] == value[1]).all(axis="columns"))[0],
            ]

    def read_info(self, attr: str, n: int, loc: int = 0) -> list:
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
        dejsonize only iff can handle dictionaries
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

    def __getitem__(self, value: tuple) -> pd.Series:
        if not isinstance(value, tuple):
            value = (value, len(self.output))
        elif len(value) < 2:
            value = (*value, len(self.output))

        param, v = value

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
    """
    COLUMNS = ["snn", "game", "results", "info"]

    def __init__(self, folder: str, filenames: list = None):
        self.folder = folder
        self.filenames = filenames if filenames is not None else os.listdir(self.folder)

        ## __a before aaa (os.listdir reads in opposite order)
        self.filenames.sort()

        self.output = None

        ## Read summary and see what columns relevant
        for i, filename in enumerate(self.filenames):
            if "SUMMARY" in filename:
                with open(os.path.join(self.folder, filename), "r") as file:
                    data = json.load(file)

                self.summary = {key: dejsonize(value) for key, value in data.items()}

                self.filenames.remove(filename)

        self.genotype_keys = list(
            self.summary["info"]["metagame_info"]["genotype_constraints"].keys()
        )

        ##
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
                    iterable = self.genotype_keys if column1 == 'info' else data[column1].keys()
                    for column2 in iterable:
                        try:
                            store.update({column2: dejsonize(data[column1][column2])})
                            continue
                        except KeyError:
                            pass

                store.update({"fitness": dejsonize(data["results"]["fitness"])})

                if self.output is None:
                    self.output = pd.DataFrame(columns=list(store.keys()))

                self.output.loc[i] = [store[key] for key in self.output.columns]
