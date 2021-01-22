"""
Delay coordinate embedding viz.

Uses matplotlib 2 or 3d.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def delay_embedding(x: np.ndarray, lag: int, dimensions: [2, 3] = 2):
    """
    Automatically infer 2D / 3D.
    """
    ## Process Data
    if dimensions == 2:
        delay_coordinates = [x[: -lag if lag else len(x)], x[lag:]]  # t-T  # t
    elif dimensions == 3:
        delay_coordinates = [
            time_series[lag : -lag if lag else len(x)],  # t-T
            time_series[2 * lag :],  # t
            time_series[: -2 * lag if lag else len(x)],  # t-2T
        ]
    else:
        raise ValueError(f"Invalid Embedding Dimension, '{dimensions}' not in [2, 3]!")

    ## Visualize Embedding
    fig = plt.figure()

    # Raw time series data
    ax = fig.add_subplot(211)
    ax.scatter(range(len(x[:100])), x[:100])
    ax.set_title("Discrete Time Series")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")

    # Reconstruction space
    ax = fig.add_subplot(212, projection="3d" if dimensions == 3 else None)
    ax.scatter(*delay_coordinates)
    ax.set_title(f"Embedding, Tau={lag}")
    ax.set_xlabel(f"x(t-{lag})")
    ax.set_ylabel("x(t)")
    if dimensions == 3:
        ax.set_zlabel(f"x(t-{2*lag})")

    plt.show()


if __name__ == "__main__":
    import csv

    with open("henon_a14b03.csv", "r") as file:
        time_series = [float(row[0]) for row in csv.reader(file)]

    delay_embedding(time_series, 1, 3)
