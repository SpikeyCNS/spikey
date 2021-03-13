"""
Convert an OpenAI gym environment into a Spikey game type.
"""
from copy import deepcopy
from spikey.games.RL.template import RL
from spikey.games.MetaRL.template import MetaRL


def gym_wrapper(env: type, base=RL) -> type:
    """
    Wrap openai gym environment for compatability within Spikey.
    Restructures environment into RL game.

    Parameters
    ----------
    env: gym.Env
        Uninitialized gym environment.
    base: type, default=RL
        Type of game to base the gym env off of.

    Return
    ------
    type(base) Restructured version of Env.
    """
    base_name = base.__name__
    type_new = deepcopy(env)

    try:
        name_new = f"{base_name}_{env.__name__}"
    except Exception:
        name_new = f"{base_name}_ENV"

    if isinstance(base, MetaRL) and not hasattr(type_new, "get_fitness"):

        def get_fitness(
            self,
            genotype: dict,
        ) -> (float, bool):
            """
            Evaluate the fitness of a genotype.
            """
            state, fitness, done, info = type_new.step(genotype)

            return fitness, done

        type_new.get_fitness = get_fitness

    type_new.__bases__ = (*type_new.__bases__, base)
    return type_new
