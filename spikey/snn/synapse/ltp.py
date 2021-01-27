"""
A synapse type updating based on all fires within window.
"""
import numpy as np

from spikey.snn.synapse.template import RLSynapse


class LTP(RLSynapse):
    """
    LTP only STDP with reward trace.

    Parameters
    ----------
    n_inputs: int
        Number of input streams.
    kwargs: dict
        Configuration dictionary. See util.get_necessary_config() for
        information on all necessary entries.
    """

    def reset(self):
        """
        Reset synapses.
        """
        self.trace = 0

    def _apply_stdp(self, full_spike_log, inhibitories):
        """
        Use stdp to update trace based on dt.

        Parameters
        ----------
        full_spike_log: np.array(time, neurons), 0 or 1
            A history of neuron firings.
        inhibitories: list[int], -1 or 1
            Neuron polarities.

        Returns
        -------
        Trace value with stdp suggestion.
        """
        ## Find how long ago each neuron fired.
        try:
            spike_log = full_spike_log[-self._stdp_window - 1 :]
        except IndexError:
            spike_log = full_spike_log

        max_time_diff = min(self._stdp_window, spike_log.shape[0])

        # spike_log = np.where(spike_log, 1, 0)

        ## Simulate decay
        decay_multiplier = np.arange(max_time_diff - 1, -1, -1).reshape((-1, 1))

        decayed_fires = decay_multiplier * spike_log

        ## Find pre and post locations
        pre_locations = np.where(np.sum(decayed_fires, axis=0))[0][:, None]
        post_locations = np.where(spike_log[-1])[0]

        if not pre_locations.size or not post_locations.size:
            return

        w_pre_locations = (
            pre_locations[np.where(pre_locations >= self._n_inputs)].reshape((-1, 1))
            - self._n_inputs
        )
        w_post_locations = (
            post_locations[np.where(post_locations >= self._n_inputs)] - self._n_inputs
        )

        update_mult = self._learning_rate / self._stdp_window * self.trace

        dts = np.where(decayed_fires, self._stdp_window + 1 - decayed_fires, 0)

        pre_dts = np.sum(dts, axis=0)[pre_locations]

        ## Apply stdp rule between every neuron-neuron pair
        if w_post_locations.size:
            self.weights._matrix[pre_locations, w_post_locations] += (
                inhibitories[pre_locations].reshape(-1, 1) * pre_dts * update_mult
            )

        np.clip(
            self.weights._matrix.data,
            0.0,
            float(self.weights._max_weight),
            out=self.weights._matrix.data,
        )

    def reward(self, rwd):
        """
        Update weights based on trace and reward.

        Parameters
        ----------
        rwd: float
            Reward the network has earned.
        """
        # self.trace += rwd
        self.trace = rwd
