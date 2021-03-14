"""
Give reward if action is the same as expected. Reward in
a spiking neural network is meant to simulate dopamine in
the real brain.
"""
import numpy as np
from spikey.module import Key
from spikey.snn.reward.template import Reward


class MatchExpected(Reward):
    """
    Give reward if action is the same as expected. Reward in
    a spiking neural network is meant to simulate dopamine in
    the real brain.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    config = {
        "reward_mult": 1,
        "punish_mult": 2,
    }
    rewarder = MatchExpected(**config)
    rewarder.reset()

    r = rewarder(state, action)
    ```

    ```python
    class network_template(Network):
        config = {
            "reward_mult": 1,
            "punish_mult": 2,
            "expected_value": ,
        }
        parts = {
            "rewarder": MatchExpected
        }
    ```
    """

    NECESSARY_KEYS = Reward.extend_keys(
        [
            Key("expected_value", "func(state)->action Expected action."),
        ]
    )

    def __call__(self, state: object, action: object) -> float:
        """
        Determine how much reward should be given for taking action in state.
        reward_mult if action == expected else punish_mult.

        Parameters
        ----------
        state: any
            Environment state before action is taken.
        action: any
            Action taken in response to state.

        Returns
        -------
        float Reward for taking action in state.
        """
        expected = self._expected_value(state)

        rwd = np.sum(
            np.where(action == expected, self._reward_mult, -self._punish_mult)
        )

        return rwd
