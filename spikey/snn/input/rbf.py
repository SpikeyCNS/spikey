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
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    processing_time = 10
    config = {
        "n_inputs": 10,
        "magnitude": 2,
        "input_firing_steps": -1,
        "input_pct_inhibitory": 0.2,
    }
    input = RBF(**config)
    input.reset()
    env = Cartpole(preset='FREMAUX')

    state = env.reset()
    for step in range(10):
        input.update(state)

        for _ in range(processing_time)
            in_fires = input.__call__()

        state, _, done, __ = env.update(0)

        if done:
            break
    ```

    ```python
    class network_template(Network):
        keys = {
            "n_inputs": 10,
            "magnitude": 2,
            "input_firing_steps": -1,
            "input_pct_inhibitory": 0.2,
        }
        parts = {
            "inputs": RBF
        }
    ```
    """

    def __call__(self) -> np.bool:
        """
        Spikes output from each input neuron.

        Returns
        -------
        ndarray[n_inputs, dtype=bool] Spike output for each neuron.
        """  ## Does not respect polarities.
        if not self.values.size:
            return []

        spikes = (
            np.int_(np.random.uniform(0, 1, size=self.values.size) <= self.values)
            * self._magnitude
        )

        self.network_time += 1
        return spikes * self.polarities

    def update(self, state: object):
        """
        Update input generator.

        Parameters
        ----------
        state: object
            Enviornment state in format generator can understand.
        """
        self.network_time = 0

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
