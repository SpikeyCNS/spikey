"""
The base Spikey Module definition. Provides structure
and functionality that every class in Spikey should
abide by.
"""
from copy import deepcopy
import pickle


class Module:
    """
    The base Spikey Module definition. Provides structure
    and functionality that every class in Spikey should
    abide by.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    class Network(Module):
        NECESSARY_KEYS = {'a': 1}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    ```

    ```python
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
    ```
    """

    NECESSARY_KEYS = {}

    def __init__(self, **kwargs):
        self._add_values(kwargs)

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

        Usage
        -----
        ```python
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
        ```
        """
        keys = deepcopy(getattr(cls, base))
        keys.update(new_keys)
        return keys

    def _add_values(self, kwargs, base="NECESSARY_KEYS", dest=None, prefix="_"):
        """
        Find all values in self.base from kwargs and add to self.

        Parameters
        ----------
        new_keys: dict
            Values to add to self.
        base: str, default="NECESSARY_KEYS"
            Name of dictionary in class to pull from.
        dest: str, default=None
            Destination to save values.
            `self if dest is None else getattr(self, dest)`
        prefix: str, default="_"
            Prefix of variables added to class with name key.

        Usage
        -----
        ```python
        Module.NECESSARY_KEYS = {'a': 'int'}
        m = Module()

        m._add_values({'a': 1}, base="NECESSARY_KEYS")

        print(m._a)  # -> 1
        ```
        """
        for key in getattr(self, base):
            setattr(
                self if dest is None else getattr(self, dest),
                f"{prefix}{key}",
                kwargs[key],
            )


def save(module: Module, filename: str, pickle_module: object=pickle, pickle_protocol=2):
    """
    Save any Module(Network, TrainingLoop, ...) to file.

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
    
    Usage
    -----
    ```python
    config = {
        "magnitude": 2,
        "n_neurons": 100,
        "neuron_pct_inhibitory": .2,
        "potential_decay": .2,
        "prob_rand_fire": .08,
        "refractory_period": 1,
        "resting_mv": 0,
        "spike_delay": 0,
    }
    neurons = Neuron(**config)

    spikey.save(synapse, 'synapse.spike')
    ```
    """
    with open(filename, 'wb') as file:
        pickle_module.dump(module, file, protocol=pickle_protocol)


def load(filename: str, pickle_module: object=pickle):
    """
    Load any Module(Network, TrainingLoop, ...) from file.
    
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

    Usage
    -----
    ```python
    synapse = spikey.load('synapse.spike')
    ```
    """
    with open(filename, 'rb') as file:
        module = pickle_module.load(file)

    if not isinstance(module, Module):
        raise ValueError("Spikey cannot load this type of file.")

    return module
