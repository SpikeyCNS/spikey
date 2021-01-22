"""
Output rate basins of attraction per input
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def out_basins(
    sim_in_rates: np.ndarray, sim_out_rates: np.ndarray, N: int = 100, title: str = ""
):
    X, Y = [], []
    for i, in_rates in enumerate(sim_in_rates):
        out_rates = sim_out_rates[i]

        if isinstance(out_rates[0], (list, tuple, np.ndarray)):
            out_rates = np.mean(out_rates, axis=1)

        in_rates = in_rates[-N:]
        out_rates = out_rates[-N:]

        for target_in in np.unique(in_rates):
            target_out = out_rates[in_rates == target_in]

            mean_out = np.mean(target_out)

            X.append(target_in)
            Y.append(mean_out)

    plt.title(title)
    sns.violinplot(x=X, y=Y)
    plt.show()
