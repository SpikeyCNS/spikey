"""
Printing / Visualizing game state info.
"""
import numpy as np


def state_transition_matrix(states: list = None, callback=None):
    """
    Generate state-state transition matrix.

    Parameters
    ----------
    states: list or ndarray of states.
    callback: ExperimentCallback with states paramater.

    Either pass states or callback, states is used by default.

    Returns
    -------
    ndarray[n_states, n_states] Transition matrix[state_i, state_i+1].
    """
    states = states or callback.info["step_states"]

    if not isinstance(states, np.ndarray) or states.dtype not in [np.int_, np.float_]:
        unique_states = list(np.unique(states))

        states = np.array([unique_states.index(value) for value in states])

    transition_matrix = np.zeros((len(unique_states), len(unique_states)))
    for i in range(len(states) - 1):
        transition_matrix[states[i], states[i + 1]] += 1

    return transition_matrix


def state_action_counts(states: list = None, actions: list = None, callback=None):
    """
    Generate state-action pair matrix with counts.

    Parameters
    ----------
    states: list or ndarray of states.
    actions: list or ndarray of actions
    callback: ExperimentCallback with states and actions paramater.

    Either pass states or callback, states is used by default.

    Returns
    -------
    ndarray[n_states, n_actions] Transition matrix[state_i, action_i].
    """
    states = states or callback.info["step_states"]
    actions = actions or callback.info["step_actions"]

    if not isinstance(states, np.ndarray) or states.dtype not in [np.int_, np.float_]:
        unique_states = list(np.unique(states))

        states = np.array([unique_states.index(value) for value in states])

    if not isinstance(actions, np.ndarray) or actions.dtype not in [np.int_, np.float_]:
        unique_actions = list(np.unique(actions))

        actions = np.array([unique_actions.index(value) for value in actions])

    pair_matrix = np.zeros((len(unique_states), len(unique_actions)))
    for i in range(len(states)):
        pair_matrix[states[i], actions[i]] += 1

    return pair_matrix


if __name__ == "__main__":
    print(state_transition_matrix(states=[1, 2, 3, 5, 6, 1, 2, 1, 2, 1, 2]))
    print(state_action_counts(states=[1, 2, 3, 5, 6, 1], actions=[1, 3, 3, 5, 6, 1]))
