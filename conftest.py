"""
Pytest collection config.

Whitelist files to collect.
"""
import os

folder_whitelist = [
    "unit_tests",
    "examples",
    "sanity",
]

file_whitelist = [
    "florian2007_XOR.ipynb",
]


def pytest_ignore_collect(path, config) -> bool:
    """
    Determine whether or not to ignore path.

    Parameters
    ----------
    path: FilePathObject
        Path to file or directory.
    config: Pytest Config Object
        Pytest config.

    Returns
    -------
    bool If path should be ignored or not.
    """
    name = str(path).split(os.sep)[-1]

    return not (
        name in folder_whitelist or name in file_whitelist or name.startswith("test")
    )
