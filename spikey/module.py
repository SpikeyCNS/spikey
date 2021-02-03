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
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

    @classmethod
    def extend_keys(cls, new_keys):
        """
        Copy the classes NECESSARY_KEYS and add new_keys.

        Parameters
        ----------
        new_keys: dict
            Keys to add to the classes NECESSARY_KEYS.

        Returns
        -------
        dict Extended version of the classes NECESSARY_KEYS.

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
        keys = deepcopy(cls.NECESSARY_KEYS)
        keys.update(new_keys)
        return keys
