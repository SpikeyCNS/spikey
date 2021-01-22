"""
Sanitizing objects for json logging.
"""
import numpy as np
from spikey.logging.serialize import compressnd


class SingleDispatch:
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


def sanitize_dictionary(dictionary):
    """
    Makes dictionary JSON safe.
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
        elif isinstance(value, ExperimentCallback):
            pass
        else:
            sanitized_dictionary[key] = sanitize(value)

    return sanitized_dictionary
