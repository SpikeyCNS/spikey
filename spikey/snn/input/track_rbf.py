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
    kwargs: dict
        Dictionary with values for each key in NECESSARY_KEYS.

    Usage
    -----
    ```python
    processing_time = 10
    config = {
        "n_inputs": 10,
        "magnitude": 2,
        "firing_steps": -1,
        "input_pct_inhibitory": 0.2,
    }
    input = TrackRBF(**config)
    env = LinearTrack()

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
        config = {
            "n_inputs": 10,
            "magnitude": 2,
            "firing_steps": -1,
            "input_pct_inhibitory": 0.2,
        }
        _template_parts = {
            "input": TrackRBF
        }
    ```
    """

    def __call__(self) -> np.ndarray:
        """
        Spikes output from each input neuron.

        Returns
        -------
        ndarray[n_inputs] Spike output for each neuron.
        """  ## Does not respect polarities.
        if not self.values.size:
            return []

        spikes = (
            np.int_(np.random.uniform(0, 1, size=self.values.size) <= self.values)
            * self._magnitude
        )

        return spikes * self.polarities

    def update(self, state: object):
        """
        Update input generator.

        Parameters
        ----------
        state: object
            Enviornment state in format generator can understand.
        """
        x, xdot, y, ydot = state

        n_x = [2 * m for m in range(-21, 22)]

        var_1 = 2

        pcm = 0.4  # 400hz

        p_t = lambda a, b: pcm * np.exp(-((x - n_x[a]) ** 2) / (var_1) ** 2)

        self.values = np.zeros((43, 5))
        for (m, n), _ in np.ndenumerate(self.values):
            self.values[m, n] = p_t(m, n)

        self.values = np.ravel(self.values)
