"""
Logic Gates
"""
import numpy as np

from spikey.games.RL.template import RL


class Logic(RL):
    """
    Mimic logic gates.

    Presets
    -------
    AND

    OR

    XOR

    NOT
    """

    action_space = [False, True]
    observation_space = [(a, b) for a in [False, True] for b in [False, True]]

    metadata = {}

    CONFIG_DESCRIPTIONS = {
        "n_inputs": "int Number of inputs.",
        "expected_value": "func(state) Expected action.",
    }

    PRESETS = {
        "AND": {
            "name": "AND",
            "n_inputs": 2,
            "expected_value": lambda state: state[0] and state[1],
        },
        "OR": {
            "name": "OR",
            "n_inputs": 2,
            "expected_value": lambda state: state[0] or state[1],
        },
        "XOR": {
            "name": "XOR",
            "n_inputs": 2,
            "expected_value": lambda state: np.sum(state) % 2,
        },
    }

    def __init__(self, preset="OR", callback=None, **kwargs):
        super().__init__(preset=preset, callback=callback, **kwargs)

    def _get_state(self):
        """
        Randomly generate a network state.
        """
        state = np.random.uniform(size=2) <= 0.5

        return tuple(state)

    def step(self, action):
        """
        Update game state and give the network info on its action.

        Returns
        -------
        State, Status
        """
        state_new = self._get_state()
        done = False

        rwd = 0
        info = {}

        self.callback.game_step(action, self._state, state_new, rwd, done, info)
        self._state = state_new
        return state_new, rwd, done, info

    def reset(self):
        """
        Initial game state.
        """
        state = self._get_state()

        self.callback.game_reset(state)
        self._state = state
        return state
