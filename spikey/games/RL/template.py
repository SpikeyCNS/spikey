"""
Base reinforcement learning environment template.
"""
import numpy as np


class RL:
    """
    Base reinforcement learning environment template.

    Parameters
    ----------
    preset: str=PRESETS.keys(), default=None
        Configuration preset key, default values for game parameters.
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to.
    kwargs: dict, default=None
        Game parameters for CONFIG_DESCRIPTIONS. Overrides preset settings.

    Usage
    -----
    ```python
    game = RL()
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

    action_space = None
    observation_space = None

    metadata = {}

    CONFIG_DESCRIPTIONS = {}
    PRESETS = {}

    def __init__(self, preset: str = None, callback: object = None, **kwargs):
        self.callback = (
            callback
            or type(
                "NotCallback",
                (object,),
                {"__getattr__": lambda s, k: lambda *a, **kw: False},
            )()
        )

        ## Generate config
        self._params = {}

        if preset is not None:
            self._params.update(self.PRESETS[preset])
        if hasattr(self, "config"):
            self._params.update(self.config)

        self._params.update(
            {key: kwargs[key] for key in self.CONFIG_DESCRIPTIONS if key in kwargs}
        )

        self._params.update({"callback": callback})

        self.callback.game_init(self)

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

        Usage
        -----
        ```python
        game = RL()
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
        # self.callback.game_step(action, state, state_new, rwd, done, info)
        raise NotImplementedError(f"step not implemented for {type(self)}")

    def reset(self) -> object:
        """
        Reset environment.

        Returns
        -------
        state Initial state.

        Usage
        -----
        ```python
        game = RL()
        game.seed(0)

        state = game.reset()
        ```
        """
        state = np.array([])

        self.callback.game_reset(state)
        return state

    def render(self, mode: str = "human"):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Example:
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
        mode (str): the mode to render with

        Usage
        -----
        ```python
        game = RL()
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.render()
        game.close()
        ```
        """
        raise NotImplementedError

    def close(self):
        """
        Shut down environment.

        Usage
        -----
        ```python
        game = RL()
        state = game.reset()

        # training loop

        game.close()
        ```
        """
        pass

    def seed(self, seed: int = None):
        """
        Seed random number generators for environment.
        """
        if seed:
            np.random.seed(seed)

        return np.random.get_state()
