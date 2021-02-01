"""
STDP runtime benchmark.
"""
import benchmarks.common as common

import numpy as np
import numpy.testing as npt

from spikey.weight import Manual
from spikey.synapse import LTP


class TimeSTDP:
    """
    Timing module.
    """

    def setup(self):
        """
        Load parameters.
        """
        N_INPUTS = 800
        N_NEURONS = 3200
        N_OUTPUTS = N_NEURONS // 2

        original_w = np.random.normal(2.5, 0.5, size=(N_INPUTS + N_NEURONS, N_NEURONS))
        original_w = np.ma.array(original_w, mask=original_w == 0)

        w_config = {
            "matrix": original_w,
            "n_inputs": N_INPUTS,
            "n_outputs": N_OUTPUTS,
            "n_neurons": N_NEURONS,
            "max_weight": 5.0,
            "inh_weight_mask": None,
        }
        s_config = {
            "stdp_window": 1000,
            "learning_rate": 0.2,
            "trace_decay": 0.1,
        }

        w = Manual(**w_config)
        self.synapse = LTP(w, **s_config, **w_config)
        self.synapse.reset()
        self.synapse._rewards = [1, 1, 1]

        spike_log = np.vstack(
            (
                np.random.uniform(
                    0, 1, size=(s_config["stdp_window"], N_INPUTS + N_NEURONS)
                )
                <= 0.33,
                np.random.uniform(0, 1, size=N_INPUTS + N_NEURONS) <= 0.33,
            )
        ).astype(np.int_)

        inh = np.where(
            np.random.uniform(0, 1, size=N_INPUTS + N_NEURONS) > 0.2, 1, -1
        ).astype(np.int_)

        self.PARAMETERS = {"main": (spike_log, inh)}

    def time_apply_stdp(self, *args):
        """
        Function to time here, may create multiple.
        """
        for _ in range(10):
            self.synapse._apply_stdp(*args)

        expected_w = common.cache_stochastic(
            "expected_w", self.synapse.weights._matrix.data
        )

        npt.assert_equal(self.synapse.weights._matrix.data, expected_w)
