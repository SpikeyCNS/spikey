"""
Parameter switch at specific time.
"""
from spikey.snn.modifier.template import Modifier


class DropOff(Modifier):
    """
    Parameter switch at specific time.

    Parameters
    ----------
    param: list
        Parameter to update, formatted as list of strings.
        eg target = network.synapse.learning_rate,
        param = ['network', 'synapse', 'learning_rate'].
    t_stop: int
        Step number to switch parameter value.
    value_start: object
        Value at start of experiment
    value_end: object
        Value at end of experiment.

    Examples
    --------

    .. code-block:: python

        modifier = DropOff('network.synapse.learning_rate'.split('.'), 2, 3, 6)
        modifier.reset()

        for step in range(100):
            modifier.update(network)
            print(network.synapse.learning_rate)  # 3 3 6 6 ...

    .. code-block:: python

        class network_template(Network):
            parts = {
                ...
                "modifiers": [
                    DropOff('network.synapse.learning_rate'.split('.'), 1, 10, 0),
                    DropOff('network.neuron.firing_threshold'.split('.'), 4, 0, 10),
                    ],
            }
    """

    def __init__(
        self, param: list, t_stop: int, value_start: object, value_end: object, *_
    ):
        super().__init__(param)

        self.t_stop = t_stop
        self.value_start = value_start
        self.value_end = value_end

        self.time = 0

    def __eq__(self, other: Modifier) -> bool:
        """
        Determine whether this modifier is the same as another.
        """
        if type(self) is not type(other):
            return False

        return all(
            [
                getattr(self, value) == getattr(other, value)
                for value in ["t_stop", "prob_start", "prob_stop"]
            ]
        )

    def reset(self):
        """
        Reset Modifier.
        Called at the start of each episode.
        """
        self.time = 0

    def update(self, network: object):
        """
        Update parameter according to rule, called once per game step.
        """
        self.time += 1

        self.set_param(
            network, self.value_start if self.time < self.t_stop else self.value_end
        )
