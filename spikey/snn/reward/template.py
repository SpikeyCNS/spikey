"""
Determine reward to give agent. Reward in a spiking neural
network is meant to simulate dopamine in the real brain.
"""


class Reward:
    """
    Determine reward to give agent. Reward in a spiking neural
    network is meant to simulate dopamine in the real brain.

    Parameters
    ----------
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    config = {
        "reward_mult": 1,
        "punish_mult": -2,
    }
    rewarder = Reward(**config)
    rewarder.reset()

    r = rewarder(state, action)
    ```

    ```python
    class network_template(Network):
        config = {
            "reward_mult": 1,
            "punish_mult": -2,
        }
        _template_parts = {
            "rewarder": Reward
        }
    ```
    """

    NECESSARY_KEYS = {
        "reward_mult": "float Multiplier for reward",
        "punish_mult": "float Multiplier for punishment.",
    }

    def __init__(self, **kwargs):
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

    def reset(self):
        """
        Reset rewarder member variables.
        """
        pass

    def __call__(self, state: object, action: object) -> float:
        """
        Determine how much reward should be given for taking action in state.

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
        raise NotImplementedError(f"__call__ not implemented for {type(self)}!")
