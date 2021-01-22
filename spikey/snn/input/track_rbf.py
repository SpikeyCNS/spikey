"""
Network input dynamics.
"""
import numpy as np

from spikey.snn.input.template import Input


class TrackRBF(Input):
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
        x, xdot, y, ydot = state

        n_x = [2 * m for m in range(-21, 22)]

        var_1 = 2

        pcm = 0.4  # 400hz

        p_t = lambda a, b: pcm * np.exp(-((x - n_x[a]) ** 2) / (var_1) ** 2)

        self.values = np.zeros((43, 5))
        for (m, n), _ in np.ndenumerate(self.values):
            self.values[m, n] = p_t(m, n)

        # import cv2
        # cv2.imshow("", self.values)
        # cv2.waitKey(1)

        self.values = np.ravel(self.values)
