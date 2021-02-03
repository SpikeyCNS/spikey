"""
The base Spikey Module definition. Provides structure
and functionality that every class in Spikey should
abide by.
"""
from copy import deepcopy


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
