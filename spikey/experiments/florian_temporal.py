"""
Temporal spike train coded XOR experiment from,
Florian R (2007) Reinforcement Learning Through Modulation of
Spike-Timing-Dependent Synaptic Plasticity. Neural Computation 19(6).
https://doi.org/10.1162/neco.2007.19.6.1468

https://www.florian.io/papers/2007_Florian_Modulated_STDP.pdf

Usage
-----
```python
from spikey.experiments.florian_temporal import (
    network_template,
    game_template,
    training_params,
)
```
"""
import numpy as np

from spikey.snn import *
from spikey.RL import *


N_INPUTS = 2
N_NEURONS = 21
N_OUTPUTS = 1

PROCESSING_TIME = 500


def get_input_map(processing_time, n_inputs):
    simple_map = {  # 100hz spike trains
        False: np.int_(
            np.random.uniform(0, 1, (processing_time, n_inputs // 2)) <= 50 * 0.0001
        ),
        True: np.int_(
            np.random.uniform(0, 1, (processing_time, n_inputs // 2)) <= 50 * 0.0001
        ),
    }

    input_map = {
        (A, B): np.hstack((simple_map[A], simple_map[B]))
        for A in [False, True]
        for B in [False, True]
    }

    return input_map


def get_w(inputs, neurons, outputs):
    w_matrix = np.vstack(
        (  # Fully connected, generated randomly over interval
            np.hstack(
                (
                    np.random.uniform(0, 0.4, (inputs, neurons - outputs)),
                    np.zeros((inputs, outputs)),
                )
            ),
            np.hstack(
                (
                    np.zeros((neurons - outputs, neurons - outputs)),
                    np.random.uniform(0, 0.4, (neurons - outputs, outputs)),
                )
            ),
            np.zeros((outputs, neurons)),
        )
    )
    w_matrix = np.ma.array(np.float16(w), mask=(w == 0), fill_value=0)

    return w_matrix


def expected_value(state):
    return np.sum(state) % 2


def continuous_rwd_action(*a):
    return True


class network_template(ContinuousRLNetwork):
    config = {
        "n_inputs": N_INPUTS,
        "n_outputs": N_OUTPUTS,
        "matrix": get_w(N_INPUTS, N_NEURONS, N_OUTPUTS),  # v/
        "n_neurons": N_NEURONS,  # v/
        "input_pct_inhibitory": 0.5,  # v/
        "neuron_pct_inhibitory": 0,  # v/
        "processing_time": PROCESSING_TIME,  # v/
        "state_spike_map": get_input_map(PROCESSING_TIME, N_INPUTS),
        "firing_threshold": 16,  # v/
        "magnitude": 1,  # v/
        "potential_decay": 0.05,  # v/ Decay constant Tau=20ms, lambda=e^(-t/T)
        "prob_rand_fire": 0.15,
        "refractory_period": 0,  # v/ Gutig, Aharonov, Rotter, & Sompolinsky 2003
        "output_range": [0, 1],  # v/
        "learning_rate": 0.25 / 25,  # v/ gamma_0 = gamma / Tau_z
        "max_weight": 5,  # v/
        "stdp_window": 20,  # v/ Tau_+ = Tau_- = 20ms
        "trace_decay": 0.04,  # v/ T_z = 25, lambda = e^(-1/T_z)
        "action_threshold": 0,  # v/ Irrelevant
        "expected_value": expected_value,
        "continuous_rwd_action": continuous_rwd_action,
    }

    _template_parts = {
        "inputs": input.StaticMap,
        "neurons": neuron.Neuron,
        "synapses": synapse.RLSTDPET,
        "weights": weight.Manual,
        "readout": readout.Threshold,
        "rewarder": reward.MatchExpected,
        "modifiers": None,
    }


class game_template(Logic):
    config = Logic.PRESETS["XOR"]


training_params = {
    "n_episodes": 1,
    "len_episode": 800,
    "eval_steps": 50,
}

__ALL__ = [network_template, game_template, training_params]
