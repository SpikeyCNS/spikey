"""
Linearly decay parameter.
"""
from spikey.snn.modifier.template import Modifier


class LinearDecay(Modifier):
    """
    Linearly decay parameter.

    Parameters
    ----------
    param: list
        Parameter to update, formatted as list of strings.
        eg target = network.synapse.learning_rate,
           param = ['network', 'synapse', 'learning_rate'].
    t_stop: int
        Time to stop decaying.
    value_start: float
        Value at start of experiment.
    value_stop: float
        Value at t_stop.

    Usage
    -----
    ```python
    modifier = LinearDecay('network.synapse.learning_rate'.split('.'), 4, 0, 3)
    modifier.reset()

    for step in range(4):
        modifier.update(network)
        print(network.synapse.learning_rate)  # 0 1 2 3
    ```

    ```python
    class network_template(Network):
        _template_parts = {
            ...
            "modifiers": [
                LinearDecay('network.synapse.learning_rate'.split('.'), 1, 0, 2),
                LinearDecay('network.neuron.firing_threshold'.split('.'), 2, 0, 4),
                ],
        }
    ```
    """

    def __init__(
        self, param: list, t_stop: int, value_start: float, value_stop: float, *_
    ):
        super().__init__(param)

        self.t_stop = t_stop
        self.change = (value_stop - value_start) / t_stop

    def __eq__(self, other: Modifier) -> bool:
        """
        Determine whether this modifier is the same as another.
        """
        if type(self) is not type(other):
            return False

        return all(
            [getattr(self, value) == getattr(other, value) for value in ["change"]]
        )

    def reset(self):
        """
        Reset Modifier.
        """
        self.time = 0

    def update(self, network: object):
        """
        Update parameter according to rule.
        """
        if self.time < self.t_stop:
            learning_rate = network.synapses._learning_rate
            self.set_param(network, learning_rate + self.change)
            self.time += 1
