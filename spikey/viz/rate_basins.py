"""
Basin of attraction of network input/output rates.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def out_basins(
    in_rates: np.ndarray, out_rates: np.ndarray, n_steps: int = None, title: str = ""
):
    """
    Violin plot to show discrete basins of attractions in network output rates
    per different network input rates.

    Parameters
    ----------
    in_rates: np.ndarray[steps, episode_len]
        Network input rates from simulator.
    out_rates: np.ndarray[steps, episode_len]
        Network output rates from simulator.
    n_steps: int, default=len(in_rates)
        Number of steps to consider.
    title: str, default=""
        Title of plot.

    Usage
    -----
    ```python
    out_basins([1, 1, 2, 2], [1, 1, 1, 2])
    ```
    """
    n_steps = n_steps or len(in_rates)
    unique_ins = np.unique(in_rates, axis=0)

    X, Y = [], []
    for i, in_rate in enumerate(in_rates):
        out_rate = out_rates[i]

        in_rate = in_rate[-n_steps:]
        out_rate = out_rate[-n_steps:]

        if isinstance(out_rate[0], (list, tuple, np.ndarray)):
            out_rate = np.mean(out_rate, axis=1)

        for target_in in unique_ins:
            target_out = out_rate[in_rate == target_in]
            mean_out = np.mean(target_out)

            if target_out.size:
                X.append(target_in)
                Y.append(mean_out)

    plt.title(title)
    sns.violinplot(x=X, y=Y)
    plt.show()
