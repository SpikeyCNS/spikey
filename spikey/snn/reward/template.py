"""
Template for reward functions.

Override
"""


class Reward:
    """
    Reward function.

    Parameters
    ----------
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    NECESSARY_KEYS = {
        "reward_mult": "float Multiplier for reward",
        "punish_mult": "float Multiplier for punishment.",
    }

    def __init__(self, **kwargs):
        for key in self.NECESSARY_KEYS:
            setattr(self, f"_{key}", kwargs[key])

    def __call__(self, state: object, action: object) -> float:
        """
        Give network a reward.

        Parameters
        ----------
        action: Readout Output
            Action taken

        Returns
        -------
        float Reward.
        """
        raise NotImplementedError(f"__call__ not implemented for {type(self)}!")
