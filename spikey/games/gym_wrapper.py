"""
Wrapper for openai gym environment.
"""
from spikey.games.RL.template import RL
from spikey.games.MetaRL.template import MetaRL


def gym_wrapper(Env: type, base=RL) -> type:
    """
    Wrap openai gym environment for compatability within Spikey.
    Restructures environment into RL game.

    Parameters
    ----------
    Env: gym.Env
        Uninitialized gym environment.
    base: type, default=RL
        Type of game to base the gym env off of.

    Return
    ------
    type(base) Restructured version of Env.
    """
    base_name = base.__name__

    try:
        name_new = f"{base_name}_{Env.__name__}"
    except Exception:
        name_new = f"{base_name}_ENV"

    if isinstance(base, MetaRL) and not hasattr(Env, 'get_fitness'):
        def get_fitness(
            self,
            genotype: dict,
            log: callable = None,
            filename: str = None,
            reduced_logging: bool = True,
            q: Queue = None,
        ) -> (float, bool):
            """
            Evaluate the fitness of a genotype.
            """
            state, fitness, done, info = Env.step(genotype)

            if q is not None:
                q.put((genotype, fitness, terminate))

            return fitness, done

        Env.get_fitness = get_fitness

    return type(name_new, (base, Env), {})