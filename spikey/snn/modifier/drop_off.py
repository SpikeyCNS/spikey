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

    def update(self, network: object):
        """
        Update parameter according to rule.
        """
        self.time += 1

        self.set_param(
            network, self.value_start if self.time < self.t_stop else self.value_end
        )
