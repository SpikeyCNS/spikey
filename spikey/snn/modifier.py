"""
Apply parameter setting function on parameter at each game step.
"""
from spikey.module import Module


class Modifier(Module):
    """
    Apply parameter setting function on parameter at each game step.

    Parameters
    ----------
    param: list
        Parameter to update, formatted as list of strings.
        eg target = network.synapse.learning_rate,
        param = ['network', 'synapse', 'learning_rate'].

    Examples
    --------

    .. code-block:: python

        modifier = Modifier('network.synapse.learning_rate'.split('.'))
        modifier.reset()

        for step in range(100):
            modifier.update(network)
            print(network.synapse.learning_rate)  # 1 2 3 4 5 ...
    """

    def __init__(self, param: list, *_):
        super().__init__(**{})
        self.param = param

    def __eq__(self, other: object) -> bool:
        """
        Determine whether this modifier is the same as another.
        """
        if type(self) is not type(other):
            return False

        params = [value for value in dir(other) if value[0] != "_"]

        return all([getattr(self, value) == getattr(other, value) for value in params])

    def reset(self):
        """
        Reset Modifier.
        Called at the start of each episode.
        """
        self.step = 0

    def set_param(self, network: object, value: float, param: list = None):
        """
        Set self._param to given value.
        """
        if param is None:
            param = self.param
        if param[0] == "network":
            obj = network

            for item in param[1:-1]:
                obj = obj.__dict__[item]

            obj.__dict__[param[-1]] = value
        else:
            raise ValueError(f"Param not recognized {param[0]}!")

    def update(self, network: object):
        """
        Update parameter according to rule, called once per game step.
        """
        self.step += 1
        raise NotImplementedError(f"Update not implemented for {type(self)}!")


class LinearDecay(Modifier):
    """
    Linearly decay parameter, updates once per game step.
    """

    def __init__(
        self, param: list, step_stop: int, value_start: float, value_stop: float, *_
    ):
        super().__init__(param)

        self.step_stop = step_stop
        self.change = (value_stop - value_start) / step_stop

    def update(self, network: object):
        """
        Update parameter according to rule, called once per game step.
        """
        if self.step < self.step_stop:
            learning_rate = network.synapses._learning_rate
            self.set_param(network, learning_rate + self.change)
            self.step += 1


class DropOff(Modifier):
    """
    Drop off parameter at certain step, updates once per game step.
    """

    def __init__(
        self, param: list, step_stop: int, value_start: object, value_end: object, *_
    ):
        super().__init__(param)

        self.step_stop = step_stop
        self.value_start = value_start
        self.value_end = value_end

    def update(self, network: object):
        """
        Update parameter according to rule, called once per game step.
        """
        self.step += 1

        self.set_param(
            network, self.value_start if self.step < self.step_stop else self.value_end
        )
