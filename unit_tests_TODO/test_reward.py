"""
Tests on the reward functions.
"""
import unittest

import numpy as np

import spikey.snn.reward as reward


class TestReward(unittest.TestCase):
    """
    Reward dynamics tests.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in [reward.MatchExpected]:
                with self.subTest(i=obj.__name__):
                    self._get_reward = self._set_obj(obj)

                    func(self)

        return run_all

    def _set_obj(self, obj):
        """
        Create generator that will render only specific object.
        """

        def _get_reward(**kwargs):
            np.random.seed(0)

            config = {
                "reward_mult": 1,
                "punish_mult": -1,
                "x_max": 1.0,
                "theta_max": 3.0,
                "n_neurons": 10,
                "n_outputs": 5,
                "time_rwd": 10,
                "expected_value": lambda *args: 1.0,
                "refractory_period": 0,
                "gamma": 0.9,
            }
            config.update(kwargs)

            get_reward = obj(**config)

            return get_reward

        return _get_reward

    @run_all_types
    def test_call(self):
        """
        Testing reward.__call__.

        Parameters
        ----------
        state: env.state
            Environment state.
        action: env action
            Environment action.
        expected: env action
            Expected action.

        Returns
        -------
        float Reward.
        """
        ## Ensure correct return type
        get_reward = self._get_reward()

        get_reward.critic_spikes = np.zeros((10))

        reward = get_reward([0, 0, 0, 0], None)

        self.assertIsInstance(
            reward, (int, np.int16, np.int32, np.int64, float, np.float32, np.float64)
        )


if __name__ == "__main__":
    unittest.main()
