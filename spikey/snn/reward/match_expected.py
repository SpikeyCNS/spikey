"""
Reward.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.reward.template import Reward


class MatchExpected(Reward):
    """
    Simple reward.
    """

    NECESSARY_KEYS = deepcopy(Reward.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {
            "expected_value": "func(state)->action Expected action.",
        }
    )

    def __call__(self, state: object, action: object) -> float:
        expected = self._expected_value(state)

        rwd = np.sum(np.where(action == expected, self._reward_mult, self._punish_mult))

        return rwd
