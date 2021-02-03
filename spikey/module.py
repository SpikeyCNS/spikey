"""
The base Spikey Module definition. Provides structure
and functionality that every class in Spikey should
abide by.
"""


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
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    ```
    """
    def __init__(self, **kwargs):
        pass
