"""
Template for RL games.
"""
import numpy as np


class RL:
    """
    Basic RL functions as a parent class to be inherited by each problem type.
    """

    action_space = None
    observation_space = None

    metadata = {}

    CONFIG_DESCRIPTIONS = {}
    PRESETS = {}

    def __init__(self, preset: str = None, callback: callable = None, **kwargs):
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

    def step(self, action) -> (object, float, bool, dict):
        """
        Returns
        -------
        state, done?
        """
        raise NotImplementedError(f"step not implemented for {type(self)}")

        # self.callback.game_step(action, state, state_new, rwd, done, info)

    def reset(self) -> object:
        """
        overwrite this function for your specific problem
        """
        state = np.array([])

        self.callback.game_reset(state)
        return state

    def render(self, mode="human"):
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
        Args:
            mode (str): the mode to render with
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
        """
        raise NotImplementedError

    def close(self):
        """
        Close environment.
        """
        pass

    def seed(self, seed=None):
        """
        Seed random number generators for environment.
        """
        if seed:
            np.random.seed(seed)

        return np.random.get_state()
