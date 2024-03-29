"""
Base reinforcement learning environment template.

Base game template. A game is the structure of an environment
that defines how agents can interact with said environment.
In this simulator they serve as an effective, modular way to
give input to and interpret feedback from the network. A game
object is not strictly required for training a network but is
highly recommended.
"""
from spikey.module import Module
import numpy as np


class RL(Module):
    """
    Base reinforcement learning environment template.

    Base game template. A game is the structure of an environment
    that defines how agents can interact with said environment.
    In this simulator they serve as an effective, modular way to
    give input to and interpret feedback from the network. A game
    object is not strictly required for training a network but is
    highly recommended.

    Parameters
    ----------
    preset: str=PRESETS.keys(), default=None
        Configuration preset key, default values for game parameters.
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.


    Examples
    --------

    .. code-block:: python

        game = RL()
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.close()

    .. code-block:: python

        class game_template(RL):
            config = RL.PRESETS["DEFAULT"]

            config.update({  # Overrides preset values
                "param1": 1
                "param2": 2,
            })

        kwargs = {
            "param1": 0,  # Overrides game_template.config["param1"]
        }
        game = game_template(**kwargs)
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.close()
    """

    action_space = None
    observation_space = None

    metadata = {}

    NECESSARY_KEYS = []
    PRESETS = {}

    def __init__(self, preset: str = None, **kwargs):
        self._params = {}
        if preset is not None:
            self._params.update(self.PRESETS[preset])
        if hasattr(self, "config"):
            self._params.update(self.config)
        self._params.update(
            {
                key.name if hasattr(key, "name") else key: kwargs[key]
                for key in self.NECESSARY_KEYS
                if key in kwargs
            }
        )
        super().__init__(**self._params)
        self._add_values(self._params, dest=self._params, prefix="")

    @property
    def params(self) -> dict:
        """
        Configuration of game.
        """
        return self._params

    def step(self, action: object) -> (object, float, bool, dict):
        """
        Act within the environment.

        Parameters
        ----------
        action: object
            Action taken in environment.

        Returns
        -------
        state: object
            Current state of environment.
        reward: float
            Reward given by environment.
        done: bool
            Whether the game is done or not.
        info: dict
            Information of environment.

        Examples
        --------

        .. code-block:: python

            game = RL()
            game.seed(0)

            state = game.reset()
            for _ in range(100):
                action = model.get_action(state)
                state, reward, done, info = game.step(action)
                if done:
                    break

            game.close()
        """
        raise NotImplementedError(f"step not implemented for {type(self)}")

    def reset(self) -> object:
        """
        Reset environment.

        Returns
        -------
        state Initial state.

        Examples
        --------

        .. code-block:: python

            game = RL()
            game.seed(0)

            state = game.reset()
        """
        raise NotImplementedError(
            f"{type(self)}.reset() not implemented! Expected to output initial state"
        )

    def render(self, mode: str = "human"):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,

        .. note::

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        .. code-block:: python

            class MyEnv(Env):
                metadata = {'render.modes': ['human', 'rgb_array']}
                def render(self, mode='human'):
                    if mode == 'rgb_array':
                        return np.array(...) # return RGB frame suitable for video
                    elif mode == 'human':
                        ... # pop up a window and render
                    else:
                        super(MyEnv, self).render(mode=mode) # just raise an exception

        Parameters
        ----------
        mode (str in ['human', 'rgb_array', 'ansi']): the mode to render with

        Examples
        --------

        .. code-block:: python

            game = Game()
            game.seed(0)

            state = game.reset()
            for _ in range(100):
                action = model.get_action(state)
                state, reward, done, info = game.step(action)
                if done:
                    break

            game.render()
            game.close()
        """
        raise NotImplementedError(f"{type(self)}.render not implemented!")

    def close(self):
        """
        Shut down environment.

        Examples
        --------

        .. code-block:: python

            game = Game()
            state = game.reset()

            # training loop

            game.close()
        """
        pass

    def seed(self, seed: int = None):
        """
        Seed random number generators for environment.
        """
        if seed:
            np.random.seed(seed)

        return np.random.get_state()
