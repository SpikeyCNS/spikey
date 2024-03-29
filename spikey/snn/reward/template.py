"""
Determine reward to give agent. Reward in a spiking neural
network is meant to simulate dopamine in the real brain.
"""
from spikey.module import Module, Key


class Reward(Module):
    """
    Determine reward to give agent. Reward in a spiking neural
    network is meant to simulate dopamine in the real brain.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Examples
    --------

    .. code-block:: python

        config = {
            "reward_mult": 1,
            "punish_mult": -2,
        }
        rewarder = Reward(**config)
        rewarder.reset()

        r = rewarder(state, action, state_next)

    .. code-block:: python

        class network_template(Network):
            keys = {
                "reward_mult": 1,
                "punish_mult": -2,
            }
            parts = {
                "rewarder": Reward
            }
    """

    NECESSARY_KEYS = [
        Key(
            "reward_mult",
            "Multiplier for reward, reward = 1 * reward_mult.",
            float,
            default=1,
        ),
        Key(
            "punish_mult",
            "Multiplier for punishment, punish = -1 * punish_mult.",
            float,
            default=0,
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._punish_mult < 0:
            print(
                "WARNING: Punish mult given is negative meaning you will give positive punishment."
            )

    def reset(self):
        """
        Reset rewarder member variables.
        Called at the start of each episode.
        """
        pass

    def __call__(self, state: object, action: object, state_next: object) -> float:
        """
        Determine how much reward should be given for taking action in state.
        Called once per game or network step based on network chosen.

        Parameters
        ----------
        state: any
            Environment state before action is taken.
        action: any
            Action taken in response to state.
        state_next: any
            State of environment after action was taken.

        Returns
        -------
        float Reward for taking action in state.
        """
        raise NotImplementedError(f"__call__ not implemented for {type(self)}!")
