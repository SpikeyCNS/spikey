"""
Game of trying to mimic logic gates.
"""
import numpy as np

from spikey.module import Key
from spikey.games.template import RL


# Top level for pickle friendliness
def and_fn(state):
    return state[0] and state[1]


def or_fn(state):
    return state[0] or state[1]


def xor_fn(state):
    return np.sum(state) % 2


class Logic(RL):
    """
    Game of trying to mimic logic gates.

    Parameters
    ----------
    preset: str=PRESETS.keys(), default="OR"
        Configuration preset key, default values for game parameters.
    callback: ExperimentCallback, default=None
        Callback to send relevant function call information to.
    kwargs: dict, default=None
        Game parameters for NECESSARY_KEYS. Overrides preset settings.


    Examples
    --------

    .. code-block:: python

        game = Logic(preset="OR")
        game.seed(0)

        state = game.reset()
        for _ in range(100):
            action = model.get_action(state)
            state, reward, done, info = game.step(action)
            if done:
                break

        game.close()

    .. code-block:: python

        class game_template(Logic):
            config = Logic.PRESETS["XOR"]

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

    action_space = [False, True]
    observation_space = [(a, b) for a in [False, True] for b in [False, True]]

    metadata = {}

    NECESSARY_KEYS = [
        Key(
            "expected_value",
            "func(state) Correct response of logic gate to specific state.",
        ),
    ]

    PRESETS = {
        "AND": {
            "name": "AND",
            "expected_value": and_fn,
        },
        "OR": {
            "name": "OR",
            "expected_value": or_fn,
        },
        "XOR": {
            "name": "XOR",
            "expected_value": xor_fn,
        },
    }

    def __init__(self, preset: str = "OR", callback: object = None, **kwargs):
        super().__init__(preset=preset, callback=callback, **kwargs)

    def _get_state(self) -> np.ndarray:
        """
        Randomly generate a network state.

        Returns
        -------
        ndarray[2, bool] Randomly generated inputs to logic gate.
        """
        state = np.random.uniform(size=2) <= 0.5

        return tuple(state)

    def step(self, action: bool) -> (np.ndarray, 0, bool, {}):
        """
        Act within the environment.

        Parameters
        ----------
        action: bool
            Action taken in environment.

        Returns
        -------
        state: np.ndarray
            ndarray[2, bool] Randomly generated inputs to logic gate.
        reward: float, = 0
            Reward given by environment.
        done: bool
            Whether the game is done or not.
        info: dict, = {}
            Information of environment.


        Examples
        --------

        .. code-block:: python

            game = Logic(preset="OR")
            game.seed(0)

            state = game.reset()
            for _ in range(100):
                action = model.get_action(state)
                state, reward, done, info = game.step(action)
                if done:
                    break

            game.close()
        """
        state_new = self._get_state()
        done = False

        rwd = 0
        info = {}

        self.callback.game_step(action, self._state, state_new, rwd, done, info)
        self._state = state_new
        return state_new, rwd, done, info

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        Returns
        -------
        np.ndarray[2, bool] Initial state, random inputs to logic gate.


        Examples
        --------

        .. code-block:: python

            game = Logic(preset="OR")
            game.seed(0)

            state = game.reset()
        """
        state = self._get_state()

        self.callback.game_reset(state)
        self._state = state
        return state
