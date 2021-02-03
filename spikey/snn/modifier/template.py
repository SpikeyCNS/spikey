"""
Parameter modifier.
"""
from spikey.module import Module


class Modifier(Module):
    """
    Parameter modifier.

    Parameters
    ----------
    param: list
        Parameter to update, formatted as list of strings.
        eg target = network.synapse.learning_rate,
           param = ['network', 'synapse', 'learning_rate'].

    Usage
    -----
    ```python
    modifier = Modifier('network.synapse.learning_rate'.split('.'))

    for step in range(100):
        modifier.update(network)
        print(network.synapse.learning_rate)  # 1 2 3 4 5 ...
    ```

    ```python
    class network_template(Network):
        _template_parts = {
            ...
            "modifiers": [
                Modifier('network.synapse.learning_rate'.split('.')),
                Modifier('network.neuron.firing_threshold'.split('.')),
                ],
        }
    ```
    """

    def __init__(self, param: list, *_):
        super().__init__(**_)
        self.param = param

    def __eq__(self, other: object) -> bool:
        """
        Determine whether this modifier is the same as another.
        """
        raise NotImplementedError(f"__eq__ not implmeneted for {type(self)}!")

    def set_param(self, network: object, value: float):
        """
        Set self._param to given value.
        """
        if self.param[0] == "network":
            obj = network

            for item in self.param[1:-1]:
                obj = obj.__dict__[item]

            obj.__dict__[self.param[-1]] = value
        else:
            raise ValueError(f"Param not recognized {self.param[0]}!")

    def update(self, network: object):
        """
        Update parameter according to rule.
        """
        raise NotImplementedError(f"Update not implemented for {type(self)}!")
