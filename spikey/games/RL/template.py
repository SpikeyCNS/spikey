"""
Base reinforcement learning environment template.

Base game template. A game is the structure of an environment
that defines how agents can interact with said environment.
In this simulator they serve as an effective, modular way to
give input to and interpret feedback from the network. A game
object is not strictly required for training a network but is
highly recommended.
"""
from spikey.games.game import Game
import numpy as np


class RL(Game):
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
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to.
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.

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

    ```python
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
    ```
    """

    action_space = None
    observation_space = None

    metadata = {}

    NECESSARY_KEYS = []
    PRESETS = {}

    def __init__(self, preset: str = None, callback: object = None, **kwargs):
        super().__init__(preset, **kwargs)

        self.callback = (
            callback
            or type(
                "NotCallback",
                (object,),
                {"__getattr__": lambda s, k: lambda *a, **kw: False},
            )()
        )
        self._params.update({"callback": callback})

        self.callback.game_init(self)

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
