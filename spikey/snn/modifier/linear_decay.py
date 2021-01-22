"""
Learning rate linear decay.
"""
from spikey.snn.modifier.template import Modifier


class LinearDecay(Modifier):
    def __init__(
        self, param: list, t_stop: int, rate_start: float, rate_stop: float, *_
    ):
        super().__init__(param)

        self.change = (rate_stop - rate_start) / t_stop

    def __eq__(self, other: Modifier) -> bool:
        """
        Primarily for genotype caching in population.
        """
        if type(self) is not type(other):
            return False

        return all(
            [getattr(self, value) == getattr(other, value) for value in ["change"]]
        )

    def update(self, network):
        """
        Update parameter once per game step.
        """
        learning_rate = network.synapses._learning_rate

        self.set_param(network, learning_rate + self.change)
