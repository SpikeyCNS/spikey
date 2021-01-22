"""
Prob rand fire drop off at certain point.
"""
from spikey.snn.modifier.template import Modifier


class DropOff(Modifier):
    def __init__(
        self, param: list, t_stop: int, prob_start: float, prob_stop: float, *_
    ):
        super().__init__(param)

        self.t_stop = t_stop
        self.prob_start = prob_start
        self.prob_stop = prob_stop

        self.time = 0

    def __eq__(self, other) -> bool:
        """
        Primarily for genotype caching in population.
        """
        if type(self) is not type(other):
            return False

        return all(
            [
                getattr(self, value) == getattr(other, value)
                for value in ["t_stop", "prob_start", "prob_stop"]
            ]
        )

    def update(self, network):
        """
        Update parameter once per game step.
        """
        self.time += 1

        self.set_param(
            network, self.prob_start if self.time < self.t_stop else self.prob_stop
        )
