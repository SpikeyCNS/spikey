"""
Viz rbf inputs.

WARNING: This is not finished and very hardcoded, waiting until RBF is more generic.
"""
import os
import numpy as np
from spikey.logging import Reader

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def cartpole_get_values(state):
    alpha = lambda a1, a2: (a1 - a2)  # % (2 * np.pi) # changed alpha!

    x, xdot, theta, thetadot = state
    lambda_thetadot = np.arctan(thetadot / 4)

    n_x = [5 / 4 * m for m in range(-3, 4)]  # 5/4 * m, m in {-3..3}
    n_xdot = [5 / 4 * n for n in range(-3, 4)]  # 5/4 * n, n in {-3..3}
    n_theta = [
        2 * np.pi / 180 * p for p in range(-7, 8)
    ]  # 2pi/3 * p - pi, p in {0..14}
    n_thetadot = [2 * np.pi / 30 * q for q in range(-7, 8)]  # 2pi/3 * q, q in {-7..7}

    var_1, var_2, var_3, var_4 = 5 / 4, 5 / 4, 1 * np.pi / 1200, 2 * np.pi / 60

    pcm = 0.4  # 400hz

    a_t = lambda a: pcm * np.exp(-((x - n_x[a]) ** 2) / (2 * var_1))
    b_t = lambda b: pcm * np.exp(-((xdot - n_xdot[b]) ** 2) / (2 * var_2))
    c_t = lambda c: pcm * np.exp(
        -alpha(theta, n_theta[c]) ** 2 / (2 * var_3)
    )  ## NOTE changed algpha
    d_t = lambda d: pcm * np.exp(
        -((lambda_thetadot - n_thetadot[d]) ** 2) / (2 * var_4)
    )

    a = np.array([a_t(i) for i in range(7)])
    b = np.array([b_t(i) for i in range(7)])
    c = np.array([c_t(i) for i in range(15)])
    d = np.array([d_t(i) for i in range(15)])

    a = np.append(a, 0.5 * np.ones(8))
    b = np.append(b, 0.5 * np.ones(8))

    a = a.reshape((1, -1))
    b = b.reshape((1, -1))
    c = c.reshape((1, -1))
    d = d.reshape((1, -1))

    return np.vstack((a, b, c, d))


def track_get_values(x):
    n_x = [2 * m for m in range(-21, 22)]

    var_1 = 2

    pcm = 0.4  # 400hz

    p_t = lambda a, b: pcm * np.exp(-((x - n_x[a]) ** 2) / (var_1) ** 2)

    values = np.zeros((43, 5))
    for (m, n), _ in np.ndenumerate(values):
        values[m, n] = p_t(m, n)

    return values


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 8))

    images = []

    """
    start_step = int(sum(ep_lens[0][:EP2]))
    finish_step = int(start_step + ep_lens[0][EP2])

    for i in range(start_step, finish_step):
        values = get_values(reader["step_states"][0][i])

        # values = get_values((0, 0, (i - 12) / 180 * np.pi, 0))

        images.append([plt.imshow(values, animated=True, cmap="gray")])
    """
    for i in range(70):
        values = track_get_values(i * 40 / 60 - 17.5)

        images.append([plt.imshow(values, animated=True, cmap="gray")])

    ani = animation.ArtistAnimation(
        fig, images, interval=300, blit=True, repeat_delay=1000
    )

    plt.show()
