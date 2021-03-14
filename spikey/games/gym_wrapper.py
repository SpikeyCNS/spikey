"""
Convert an OpenAI gym environment into a Spikey game type.

Usage
-----
```python
from gym.envs.classic_control import cartpole
cartpole_env = gym_wrapper(cartpole.CartPoleEnv, base=RL.template.RL)

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
```
"""
from copy import deepcopy
from spikey.games.game import Game
from spikey.games.RL.template import RL
from spikey.games.MetaRL.template import MetaRL


class GymWrap(Game):
    # Use by adding 2 classes to __bases__,
    # a Game derivative then a gym env.
    def __init__(self, env_kwargs={}, *args, **kwargs):
        mro = type(self).__mro__
        game_idx = mro.index(Game)
        super().__init__(**env_kwargs)  # Always env regardless of its MRO
        mro[game_idx - 1].__init__(
            self, *args, **kwargs
        )  # base, asserting base.__base__ == Game

    def get_fitness(
        self,
        genotype: dict,
    ) -> (float, bool):
        """
        Evaluate the fitness of a genotype.
        """
        state, fitness, done, info = self.step(genotype)
        return fitness, done


def gym_wrapper(env: type, base=RL) -> type:
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
    base: type, default=RL
        Type of game to base the gym env off of.

    Return
    ------
    GymWrap Restructured version of Env. Notably, if need to pass
    any parameters to the gym env, do GymWrap(env_kwargs={...}, **RL_kwargs)

    Usage
    -----
    ```python
    from gym.envs.classic_control import cartpole
    cartpole_env = gym_wrapper(cartpole.CartPoleEnv, base=RL.template.RL)

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
    ```
    """
    type_new = deepcopy(GymWrap)
    type_new.__bases__ = (env, base)

    base_name = base.__name__
    try:
        name_new = f"{base_name}_{env.__name__}"
    except Exception:
        name_new = f"{base_name}_ENV"
    type_new.__name__ = name_new

    return type_new
