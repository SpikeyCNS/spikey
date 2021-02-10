"""
Sanitizing objects for json logging.
"""
import json
import numpy as np
from spikey.logging.serialize import compressnd
from spikey.module import Module


class SingleDispatch:
    """
    Apply call operator according to parameter type.

    Parameters
    ----------
    fn: callable
        lambda x: object Default function.

    Usage
    -----
    ```python
    @SingleDispatch
    def base_fn(x: object) -> object:
        return x

    @base_fn.register(int)
    def base_fn_int(x: int) -> object
        return x + 1

    if __name__ == '__main__':
        print(base_fn("test"))  # -> "test"
        print(base_fn(2))  # -> 3
    ```
    """

    def __init__(self, fn: callable):
        self.default = fn

        self._type_mapping = {}

    def __call__(self, x: str = None) -> object:
        for key, fn in self._type_mapping.items():
            if isinstance(x, key):
                return fn(x)

        return self.default(x)

    def register(self, *args) -> callable:
        def register_fn(fn):
            self._type_mapping.update({arg: fn for arg in args})

            return self

        return register_fn


@SingleDispatch
def sanitize(value: object) -> object:
    """
    Default sanitize.
    """
    return value


@sanitize.register(np.integer)
def a(value):
    return int(value)


@sanitize.register(np.float32, np.float64)
def b(value):
    return float(value)


@sanitize.register(list, np.ndarray)
def c(value):
    return compressnd(value)


@sanitize.register(np.ma.MaskedArray)
def d(value):
    return compressnd(value.data)


@sanitize.register(dict)
def e(value):
    output = {}
    for key, v in value.items():
        if callable(v):
            continue

        output[str(key)] = sanitize(v)

    return output


def sanitize_dictionary(dictionary: dict) -> dict:
    """
    Makes dictionary JSON safe.

    Effects
    -------
    * Applies recursively.
    * Removes callbacks.
    * list, ndarray -> single line string via compressnd.
    * ma.ndarray -> Removes mask.
    * All else preserved.

    Parameters
    ----------
    dictionary: dict
        Dictionary to sanitize.

    Returns
    -------
    dict Json safe dictionary.

    Usage
    -----
    ```python
    sanitized = sanitize_dictionary({'a': np.ones(3)})
    json.dump(sanitized)
    ```
    """
    # circular import fix
    from spikey.core import ExperimentCallback

    sanitized_dictionary = {}
    for key, value in dictionary.items():
        if callable(value):
            continue

        if isinstance(key, tuple):
            key = str(key)
        elif isinstance(key, np.integer):
            key = int(key)
        elif isinstance(key, (np.float32, np.float64)):
            key = float(key)

        if isinstance(value, dict):
            sanitized_dictionary[key] = sanitize_dictionary(value)
        elif isinstance(value, tuple):
            sanitized_dictionary[key] = tuple([sanitize(v) for v in value])
        elif isinstance(value, (ExperimentCallback, Module)):
            sanitized_dictionary[key] = value.__name__ if hasattr(value, '__name__') else None
        else:
            try:
                value = sanitize(value)
                json.dumps(value)
            except:
                value = None
            sanitized_dictionary[key] = value

    return sanitized_dictionary
