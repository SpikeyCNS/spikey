"""
Wrapper for openai gym environment.
"""
from spikey.games.RL.template import RL


def gym_wrapper(Env: type) -> type:
    """
    Wrap openai gym environment for compatability within Spikey.
    Restructures environment into RL game.

    Parameters
    ----------
    Env: gym.Env
        Uninitialized gym environment.

    Return
    ------
    RL Restructured version of Env.
    """
    try:
        name_new = f"RL_{Env.__name__}"
    except Exception:
        name_new = "RL_ENV"

    return type(name_new, (RL, Env), {})
