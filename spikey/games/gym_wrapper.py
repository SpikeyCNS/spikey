"""
Convert an OpenAI gym environment into a Spikey game.

Examples
--------

.. code-block:: python

    from gym.envs.classic_control import cartpole
    cartpole_env = gym_wrapper(cartpole.CartPoleEnv)

    kwargs = {
        "param1": 0,
    }
    game = cartpole_env(**kwargs)
    game.seed(0)

    state = game.reset()
    for _ in range(100):
        action = model.get_action(state)
        state, reward, done, info = game.step(action)
        if done:
            break

    game.close()
"""
from copy import deepcopy
from spikey.games.template import RL


class GymWrap(RL):
    # Use by adding 2 classes to __bases__,
    # a Game derivative then a gym env.
    def __init__(self, env_kwargs={}, *args, **kwargs):
        mro = type(self).__mro__
        game_idx = mro.index(RL)
        super().__init__(**env_kwargs)  # Always env regardless of its MRO
        mro[game_idx - 1].__init__(
            self, *args, **kwargs
        )  # base, asserting base.__base__ == RL


def gym_wrapper(env: type) -> type:
    """
    Wrap openai gym environment for compatability within Spikey.
    Restructures environment into RL game.

    WARNING: May break inheretance when wrapping multiple different
    gym envs in the same file, check the wrapped_env.__mro__ of each
    to ensure has only desired values.

    Parameters
    ----------
    env: gym.Env
        Uninitialized gym environment.

    Return
    ------
    GymWrap Restructured version of Env. Notably, if need to pass
    any parameters to the gym env, do GymWrap(env_kwargs={...}, **RL_kwargs)

    Examples
    --------

    .. code-block:: python

        from gym.envs.classic_control import cartpole
        cartpole_env = gym_wrapper(cartpole.CartPoleEnv)

        gym_kwargs = {

        }
        kwargs = {
            "param1": 0,
        }
        game = cartpole_env(env_kwargs=gym_kwargs, **kwargs)
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.close()
    """
    type_new = deepcopy(GymWrap)
    type_new.__bases__ = (env, RL)

    try:
        name_new = f"RL_{env.__name__}"
    except Exception:
        name_new = f"RL_ENV"
    type_new.__name__ = name_new

    return type_new
