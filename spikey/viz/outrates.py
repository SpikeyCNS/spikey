"""
Scatter plot input_rates vs output_rate.
"""
import numpy as np
import matplotlib.pyplot as plt


def outrates_scatter(
    ins: np.ndarray, outs: np.ndarray, N: int = None, titles: list = None
):
    """
    Scatter plates of output rates vs input rates.

    Parameters
    ----------
    ins: np.ndarray[steps, episode_len]
        Network input rates from simulator.
    outs: np.ndarray[steps, episode_len]
        Network output rates from simulator.
    N: int, default=len(ins)
        Number of steps to consider. Generates new plot per each step.
    titles: str, default=""
        Title per each scatter plot.

    Usage
    -----
    ```python
    outrates_scatter(info['input_rates'], info['output_rates'])
    ```
    """
    length_episode = len(ins[0])
    N = N or len(ins)

    for i in range(len(ins) - N, len(ins)):
        inrates = ins[i]
        outrates = outs[i]

        if isinstance(outrates[0], (tuple, list, np.ndarray)):
            plt.scatter(inrates[-length_episode:], outrates[-length_episode:, 0])
            plt.scatter(inrates[-length_episode:], outrates[-length_episode:, 1], s=10)
        else:
            plt.scatter(inrates[-length_episode:], outrates[-length_episode:])

        plt.xlabel("input rates")
        plt.ylabel("output rates")

        if titles is not None:
            plt.title(titles[i])

        plt.show()
