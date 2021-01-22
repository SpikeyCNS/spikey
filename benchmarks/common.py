"""
Common functions for benchmarks to make use of.

NOTE: Do no edit, only append!
"""
import numpy as np


def cache_stochastic(key: str, value: np.ndarray, reset: bool = False, dtype=None):
    """
    Store stochastic variables.
    """
    filename = f"{key}.dat"
    output = None

    if not reset:
        try:
            output = np.fromfile(filename, dtype=dtype).reshape(value.shape)
        except FileNotFoundError:
            pass

    if output is None:
        output = value
        value.tofile(filename)

    return output
