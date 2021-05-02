"""
Tests for snn.Reward.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import reward


def expected_state(state):
    return state


class TestReward(unittest.TestCase, ModuleTest):
    """
    Tests for snn.Reward.
    """

    TYPES = [reward.MatchExpected]
    BASE_CONFIG = {
        "reward_mult": 1,
        "punish_mult": 1,
        "expected_value": expected_state,
    }

    @ModuleTest.run_all_types
    def test_usage(self):
        rewarder = self.get_obj()
        rewarder.reset()

        r = rewarder(1, 0, 0)


if __name__ == "__main__":
    unittest.main()
