"""
Vizualize state-output rates.
"""
import numpy as np
import matplotlib.pyplot as plt


def outrates_scatter(
    ins: np.ndarray, outs: np.ndarray, length_episode: int, N: int, titles: list = None
):
    for i in range(N):
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
