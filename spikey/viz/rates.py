"""
Viz neuron firing rates.
"""
from string import ascii_uppercase
import numpy as np


def print_rates(
    step_inrates: np.float = None,
    step_outrates: np.float = None,
    step_states: np.ndarray = None,
    observation_space: list = None,
    callback: object = None,
    precision=2,
    episode=-1,
):
    """
    Print the input and output rate in response to each state

    Parameters
    ----------
    step_inrates: [[ep1step1_inrate, ...], [ep2step1_inrate, ...]]
        Input neuron firing rates.
    step_outrates: [[ep1step1_outrate, ...], [ep2step1_outrate, ...]]
        Output neuron firing rates.
    step_states: [[ep1step1_state, ...], [ep2step1_state, ...]]
        Game state at each step.
    observation_space: list
        Game observation space.
    callback: ExperimentCallback, default=None
        Callback to fill all parameters if not explicitly given.
    precision: int
        Number of decimal places to print.
    episode: int or [int, ..] or None(all), default=-1
        Episode or episodes to average rates over.

    Examples
    --------

    .. code-block:: python

        print_rates(callback=callback)

        # state1: inrates[step_states==state1] -> outrates[step_states==state1]
        # state2: inrates[step_states==state2] -> outrates[step_states==state2]
        # ...

    """
    step_inrates = step_inrates or callback.info["step_inrates"]
    step_outrates = step_outrates or callback.info["step_outrates"]
    step_states = step_states or callback.info["step_states"]
    observation_space = observation_space or callback.game.observation_space

    if episode is not None:
        step_inrates = np.array(step_inrates)[episode]
        step_outrates = np.array(step_outrates)[episode]
        step_states = np.array(step_states)[episode]

    for state in observation_space:
        state_mask = step_states == state
        if step_states.ndim > step_inrates.ndim:
            state_mask = state_mask.all(axis=-1)
        mean_inrates = np.mean(step_inrates[state_mask])
        mean_outrates = np.mean(step_outrates[state_mask])
        print(f"{state}: {mean_inrates:.{precision}f} -> {mean_outrates:.{precision}f}")


def print_common_action(
    step_inrates: np.float = None,
    step_actions: np.ndarray = None,
    step_states: np.ndarray = None,
    observation_space: list = None,
    callback: object = None,
    episode=-1,
):
    """
    Print the most common action taken in response to each state.

    Parameters
    ----------
    step_inrates: [[ep1step1_inrate, ...], [ep2step1_inrate, ...]]
        Input neuron firing rates.
    step_actions: [[ep1step1_action, ...], [ep2step1_action, ...]]
        Network action at each step.
    step_states: [[ep1step1_state, ...], [ep2step1_state, ...]]
        Game state at each step.
    observation_space: list
        Game observation space.
    callback: ExperimentCallback, default=None
        Callback to fill all parameters if not explicitly given.
    episode: int or [int, ..] or None(all), default=-1
        Episode or episodes to average rates over.

    Examples
    --------

    .. code-block:: python

        print_max_action(callback=callback)

        # state1: inrates[step_states==state1] -> most_common(actions[step_states==state1])
        # state2: inrates[step_states==state2] -> most_common(actions[step_states==state2])
        # ...
    """
    step_inrates = step_inrates or callback.info["step_inrates"]
    step_actions = step_actions or callback.info["step_actions"]
    step_states = step_states or callback.info["step_states"]
    observation_space = observation_space or callback.game.observation_space

    if episode is not None:
        step_inrates = np.array(step_inrates)[episode]
        step_actions = np.array(step_actions)[episode]
        step_states = np.array(step_states)[episode]

    for state in observation_space:
        state_mask = step_states == state
        mean_inrates = np.mean(step_inrates[state_mask])
        actions, counts = np.unique(step_actions[state_mask], return_counts=True)
        action = actions[np.argmax(counts)]
        print(
            f"{state}: {mean_inrates:.4f} -> {ascii_uppercase[action]}({action}). counts={counts}"
        )
