"""
Wrap an openai gym env, making it inheret RL.
"""
from spikey.games.RL.template import RL


def gym_wrapper(Env: type) -> type:
    """
    Wrap an openai gym env, make it inheret RL.

    Parameters
    ----------
    Env: Class
        An openai gym environment class(not initialized!).

    Return
    ------
    class Updated version of env that inherets RL.
    """
    try:
        name_new = f"RL_{Env.__name__}"
    except Exception:
        name_new = "RL_ENV"

    return type(name_new, (RL, Env), {})
