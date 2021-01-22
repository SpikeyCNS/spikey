"""
Network input dynamics.
"""
import numpy as np

from spikey.snn.input.template import Input


class RBF(Input):
    """
    Radial basis function neurons to simulate place cells.

    Parameters
    ----------
    n_inputs: int
        Number of inputs.
    """

    def __call__(self):
        """
        Spike output for each input neuron.

        Returns
        -------
        ndarray Spike output for each neuron.
        """  ## Does not respect polarities.
        if not self.values.size:
            return []

        spikes = (
            np.int_(np.random.uniform(0, 1, size=self.values.size) <= self.values)
            * self._magnitude
        )

        return spikes * self.polarities

    def update(self, state):
        """
        Update input settings.

        Parameters
        ----------
        state: list of float
            Discretized enviornment state.
        """
        alpha = lambda a1, a2: (a1 - a2)  # % (2 * np.pi)

        x, xdot, theta, thetadot = state
        lambda_thetadot = np.arctan(thetadot / 4)

        n_x = [5 / 4 * m for m in range(-3, 4)]  # 5/4 * m, m in {-3..3}
        n_xdot = [5 / 4 * n for n in range(-3, 4)]  # 5/4 * n, n in {-3..3}
        n_theta = [
            2 * np.pi / 180 * p for p in range(-7, 8)
        ]  # 2pi/3 * p - pi, p in {0..14}
        n_thetadot = [
            2 * np.pi / 30 * q for q in range(-7, 8)
        ]  # 2pi/3 * q, q in {-7..7}

        var_1, var_2, var_3, var_4 = 5 / 4, 5 / 4, 1 * np.pi / 1200, 2 * np.pi / 60

        pcm = 0.4  # 400hz

        p_t = lambda a, b, c, d: pcm * np.exp(
            -((x - n_x[a]) ** 2)
            / (2 * var_1)  #    -(xdot-n_xdot[b])**2 / (2 * var_2) \
            - alpha(theta, n_theta[c]) ** 2 / (2 * var_3)
            - (lambda_thetadot - n_thetadot[d]) ** 2 / (2 * var_4)
        )

        self.values = np.zeros((7, 15, 15))
        for (m, p, q), _ in np.ndenumerate(self.values):
            n = 0
            self.values[m, p, q] = p_t(m, n, p, q)

        self.values = np.ravel(self.values)
