"""
Hard threshold population readout.
"""
from copy import deepcopy
import numpy as np

from spikey.snn.readout.template import Readout


class Threshold(Readout):
    """
    Threshold population readout.
    """

    NECESSARY_KEYS = deepcopy(Readout.NECESSARY_KEYS)
    NECESSARY_KEYS.update(
        {"action_threshold": "float or 'mean' Threshold to trigger high state."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._action_threshold == "mean":
            self.rate_log = []

    def __call__(self, output_spike_train: np.bool) -> object:
        if self._n_outputs == 0:
            return 0

        rate = np.mean(output_spike_train) / self._magnitude
        if self._action_threshold == "mean":
            threshold = np.mean(self.rate_log) if self.rate_log else 0
            self.rate_log.append(rate)
        else:
            threshold = self._action_threshold
        action = self._output_range[rate >= threshold]

        return action
