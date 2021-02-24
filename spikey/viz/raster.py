"""
Spike raster plot.
"""
import numpy as np
import matplotlib.pyplot as plt


def spike_raster(
    spike_log: np.bool = None,
    rewards: np.float = None,
    polarities: np.int = None,
    callback: object = None,
):
    """
    Render a spike raster plot.
    Fires are white, inhibitories tinted red.

    Parameters
    ----------
    spike_log: ndarray[steps, neurons]
        Spike train.
    rewards: ndarray[steps]
        Rewards at each step.
    polarities: ndarray[neurons]
        Polaritiy of(optionally inputs and) neurons.
    network: Network, default=None
        Network to give spike_log and polarities if not explicitly given.
    callback: ExperimentCallback, default=None
        Callback to give rewards if not explicitly given.

    Usage
    -----
    ```python
    spike_raster(network, game)
    ```
    """
    try:
        spike_log = spike_log or network.spike_log
    except:
        raise UnboundLocalError(
            "Need to define either spike_log or network for raster!"
        )
    polarities = polarities or network._polarities
    rewards = rewards or callback.info["step_rewards"]

    spike_log = np.abs(spike_log).T
    spike_log = np.dstack((spike_log, spike_log, spike_log))

    if polarities is not None:
        inh_locs = np.where(polarities == -1)[0]
        if len(polarities == len(spike_log)):
            spike_log[inh_locs, :, 0] = 0.5
        else:
            spike_log[n_inputs:][inh_locs, :, 0] = 0.5
        spike_log = np.abs(spike_log)  # Fix weird bug setting b,g = -1

    if rewards is not None:
        reward_helper = rewards.reshape((-1, 1))
        reward_helper = np.ravel(np.hstack([reward_helper for _ in range(step_len)]))
        rewards = np.zeros(shape=(reward_helper.size, 3))

        rewards[np.where(reward_helper < 0)] = np.array([1, 0, 0])
        rewards[np.where(reward_helper == 0)] = np.array([0, 0, 0])
        rewards[np.where(reward_helper > 0)] = np.array([0, 1, 0])

        if len(rewards.shape) < len(spike_log.shape):
            rewards = rewards.reshape((1, *rewards.shape))

        output = np.vstack((spike_log, rewards))

    fig = plt.figure()
    plt.gcf().canvas.set_window_title(f"Spike Raster")

    ax = fig.add_subplot(111)
    ax.set_title(f"Raster{' w/ Rewards' if rewards is not None else ''}")
    ax.imshow(output)

    plt.show()
