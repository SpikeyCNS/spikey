"""
The base Spikey Module definition. Provides structure
and functionality that every class in Spikey should
abide by.
"""
from copy import deepcopy
import pickle
import numpy as np


class Key:
    """
    A parameter for some module part.

    Parameters
    ----------
    name: str
        Name / key of the parameter.
    description: str
        Explination of the purpose of the parameter.
    type: type, default=any
        Required type of the parameter or types in a tuple.
    default: object, default=N/A
        Default value if no value is given for this key.

    Examples
    --------

    .. code-block:: python

        class X(Module):
            NECESSARY_KEYS = [
                Key('name', "description", type, default_value),
            ]

        x.list_keys()
    """

    def __init__(self, name, description, type=any, default="veryspecificstring"):
        self.name = name
        self.type = type
        if self.type == float:
            self.type = (float, int)
        self.description = description
        if default != "veryspecificstring":
            self.default = default

    def __str__(self):
        t = self.type if self.type != any else "any"
        default_str = f", default={self.default}" if hasattr(self, "default") else ""
        return f'"{self.name}": "[{t}{default_str}] {self.description}"'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Key):
            return self.name == other.name
        return self.name == other

    def __hash__(self):
        return hash(self.name)


class Module:
    """
    The base Spikey Module definition. Provides structure
    and functionality that every class in Spikey should
    abide by.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        class Network(Module):
            NECESSARY_KEYS = [Key('a', 'basic parameter', type=int, default=100)

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        Network.list_keys()

    .. code-block:: python

        class Network(Module):
            NECESSARY_KEYS = {'a': 1}

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        Network.list_keys()

    .. code-block:: python

        class Network(Module):
            NECESSARY_KEYS = [
                Key('a', 'desc')
            ]

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class RLNetwork(Network):
            NECESSARY_KEYS = Network.extend_keys([
                Key('b', 'desc')
            ])

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        RLNetwork.list_keys()

    .. code-block:: python

        class Network(Module):
            NECESSARY_KEYS = {'a': 1}

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class RLNetwork(Network):
            NECESSARY_KEYS = Network.extend_keys({
                'b': 2,
            })

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        RLNetwork.list_keys()
    """

    NECESSARY_KEYS = []

    def __init__(self, **kwargs):
        self._add_values(kwargs)
        self.training = True

    @classmethod
    def extend_keys(cls, new_keys, base="NECESSARY_KEYS"):
        """
        Copy the class.base and add new_keys.

        Parameters
        ----------
        new_keys: dict
            Keys to add to cls.base.
        base: str, default="NECESSARY_KEYS"
            Name of dictionary in class to extend.

        Returns
        -------
        dict Extended version of the class.base.

        Examples
        --------

        .. code-block:: python

            class Network(Module):
                NECESSARY_KEYS = [
                    Key('a', 'desc')
                ]

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            class RLNetwork(Network):
                NECESSARY_KEYS = Network.extend_keys([
                    Key('b', 'desc')
                ])

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

        .. code-block:: python

            class Network(Module):
                NECESSARY_KEYS = {'a': 1}

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            class RLNetwork(Network):
                NECESSARY_KEYS = Network.extend_keys({
                    'b': 2,
                })

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
        """
        keys = getattr(cls, base)
        keys = deepcopy(keys)

        if isinstance(keys, dict):
            if isinstance(new_keys, list):
                new_keys = {key.name: key.description for key in new_keys}
            keys.update(new_keys)
        elif isinstance(keys, list):
            if isinstance(new_keys, dict):
                new_keys = [
                    Key(key, description) for key, description in new_keys.items()
                ]
            keys.extend(new_keys)

        return keys

    @classmethod
    def list_keys(cls):
        """
        Print list of all required keys for this Module.

        Examples
        --------

        .. code-block:: python

            class Network(Module):
                NECESSARY_KEYS = [Key('a', 'basic parameter', type=int, default=100)

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            Network.list_keys()

        .. code-block:: python

            class Network(Module):
                NECESSARY_KEYS = {'a': 1}

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            Network.list_keys()
        """
        print("{")
        for key in cls.NECESSARY_KEYS:
            if isinstance(key, Key):
                print(f"\t{str(key)},")
            else:
                desc = cls.NECESSARY_KEYS[key]
                print(f"\t{key}: {desc},")

        print("}")

    def train(self):
        """
        Set the module to training mode, enabled by default.
        """
        self.training = True

    def eval(self):
        """
        Set the module to evaluation mode, disabled by default.
        """
        self.training = False

    def copy(self) -> object:
        """
        Return a deepcopy of self.
        """
        return deepcopy(self)

    def _check_config(self, kwargs, base="NECESSARY_KEYS"):
        """
        Ensure all necessary keys from base are in kwargs.

        Parameters
        ----------
        kwargs: dict
            Values to add to self.
        base: str, default="NECESSARY_KEYS"
            Name of dictionary in class to pull from.

        Raises
        ------
        KeyError if any expected keys are not present.
        """
        missing = []
        for key in getattr(self, base):
            if isinstance(key, Key):
                name = key.name
                if name not in kwargs:
                    if not hasattr(key, "default"):
                        missing.append(name)

            elif isinstance(key, str):
                if key not in kwargs:
                    missing.append(key)

        if len(missing):
            raise KeyError(
                f"Missing values for keys {missing}, all of wich do not have defaults!"
            )

    def _add_values(self, kwargs, base="NECESSARY_KEYS", dest=None, prefix="_"):
        """
        Find all values in self.base from kwargs and add to self.

        Parameters
        ----------
        kwargs: dict
            Values to add to self.
        base: str, default="NECESSARY_KEYS"
            Name of dictionary in class to pull from.
        dest: str, default=None
            Destination to save values.
            `self if dest is None else getattr(self, dest)`
        prefix: str, default="_"
            Prefix of variables added to class with name key.

        Examples
        --------

        .. code-block:: python

            Module.NECESSARY_KEYS = [Key('a', 'description')]
            m = Module()

            m._add_values({'a': 1}, base="NECESSARY_KEYS")

            print(m._a)  # -> 1

        .. code-block:: python

            Module.NECESSARY_KEYS = {'a': 'int'}
            m = Module()

            m._add_values({'a': 1}, base="NECESSARY_KEYS")

            print(m._a)  # -> 1
        """
        dest = dest or self
        if isinstance(dest, str):
            dest = getattr(self, dest)
        self._check_config(kwargs, base)

        for key in getattr(self, base):
            if isinstance(key, Key):
                name = key.name
                if name in kwargs:
                    value = kwargs[name]
                    if key.type == np.array:
                        key.type = np.ndarray
                    if key.type != any and not isinstance(value, key.type):
                        if key.type is np.ndarray and isinstance(value, (list, tuple)):
                            value = np.array(value)
                        else:
                            raise KeyError(
                                f"Key {name} is incorrect type, got {type(value)} and expected {key.type}!"
                            )
                else:
                    if not hasattr(key, "default"):
                        raise KeyError(f"No value given for key, '{name}'!")

                    value = key.default

            elif isinstance(key, str):
                name = key
                value = kwargs[name]

            if isinstance(dest, dict):
                dest[f"{prefix}{name}"] = value
            else:
                setattr(dest, f"{prefix}{name}", value)


def save(
    module: Module, filename: str, pickle_module: object = pickle, pickle_protocol=2
):
    """
    Save any Module(Network, ...) to file.

    Parameters
    ----------
    module: Module
        Module to save.
    filename: str
        Filename to save module to.
    pickle_module: python package, default=pickle
        Python package that defines how data will be saved.
    pickle_protocol: int, default=2
        Saving protocol for pickle.

    Examples
    --------

    .. code-block:: python

        config = {
            "magnitude": 2,
            "n_neurons": 100,
            "neuron_pct_inhibitory": .2,
            "potential_decay": .2,
            "prob_rand_fire": .08,
            "refractory_period": 1,
        }
        neurons = Neuron(**config)

        spikey.save(synapse, 'synapse.spike')
    """
    with open(filename, "wb") as file:
        pickle_module.dump(module, file, protocol=pickle_protocol)


def load(filename: str, pickle_module: object = pickle):
    """
    Load any Module(Network, ...) from file.

    NOTE: loading pickle files is inherently insecure given
    you are directly loading arbitrary objects into python.
    Only load files from sources you trust.
    https://security.stackexchange.com/questions/183966/safely-load-a-pickle-file

    Parameters
    ----------
    filename: str
        Filename to load module from.
    pickle_module: python package, default=pickle
        Python package that defines how data will be loaded.

    Returns
    -------
    Module Module pulled from file.

    Examples
    --------

    .. code-block:: python

        synapse = spikey.load('synapse.spike')
    """
    with open(filename, "rb") as file:
        module = pickle_module.load(file)

    if not isinstance(module, Module):
        raise ValueError("Spikey cannot load this type of file.")

    return module
