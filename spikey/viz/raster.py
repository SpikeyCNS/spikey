"""
Spike raster plotting.
"""
import numpy as np
import matplotlib.pyplot as plt


def spike_raster(
    spikes: np.ndarray,
    steps: int,
    step_len: int,
    rewards: np.ndarray = None,
    input_polarities: np.ndarray = None,
    polarities: np.ndarray = None,
    n_inputs: int = None,
):
    """
    Render a spike raster plot.

    Parameters
    ----------
    spikes: 2d ndarray
        Spike train.
    steps: int
        Number of game steps to plot.
    step_len: int
        Number of network iterations per step.
    rewards: ndarray
        Rewards at each step.
    polarities: ndarray
        Polaritiy of neurons.
    n_inputs: int
        Number of input neurons, serves as offset for polarities.
    """
    output = None

    ## Spikes
    # fire is white, inhibitories tinted red
    spikes = np.abs(spikes[-(steps * step_len) :]).T
    spikes = np.dstack((spikes, spikes, spikes))

    output = spikes

    if input_polarities is not None:
        inh_locs = np.where(input_polarities == -1)[0]
        spikes[:n_inputs][inh_locs, :, 0] = 0.5
        spikes = np.abs(spikes)  # Fix weird bug setting b,g = -1

    if polarities is not None:
        if n_inputs is None:
            print("Warning: May need n_inputs for polarities to be placed correctly!")

        inh_locs = np.where(polarities == -1)[0]
        spikes[n_inputs:][inh_locs, :, 0] = 0.5
        spikes = np.abs(spikes)  # Fix weird bug setting b,g = -1

    if rewards is not None:
        ## Rewards
        # - is red, 0 is black, + is green
        rewards = np.array(rewards)

        reward_helper = rewards[-steps:].reshape((-1, 1))

        reward_helper = np.ravel(np.hstack([reward_helper for _ in range(step_len)]))

        rewards = np.zeros(shape=(reward_helper.size, 3))

        rewards[np.where(reward_helper < 0)] = np.array([1, 0, 0])
        rewards[np.where(reward_helper == 0)] = np.array([0, 0, 0])
        rewards[np.where(reward_helper > 0)] = np.array([0, 1, 0])

        if len(rewards.shape) < len(spikes.shape):
            rewards = rewards.reshape((1, *rewards.shape))

        output = np.vstack((spikes, rewards))

    #
    fig = plt.figure()
    plt.gcf().canvas.set_window_title(f"Spike Raster Plot, past {steps} steps")

    ax = fig.add_subplot(111)
    ax.set_title(f"Raster{' w/ Rewards' if rewards is not None else ''}")
    ax.imshow(output)

    plt.show()
