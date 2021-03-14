"""
Rate based XOR experiment from,
Florian R (2007) Reinforcement Learning Through Modulation of
Spike-Timing-Dependent Synaptic Plasticity. Neural Computation 19(6).
https://doi.org/10.1162/neco.2007.19.6.1468

https://www.florian.io/papers/2007_Florian_Modulated_STDP.pdf

Usage
-----
```python
from spikey.experiments.florian_rate import (
    network_template,
    game_template,
    training_params,
)
```
"""
import numpy as np

from spikey.snn import *
from spikey.RL import *


N_INPUTS = 60
N_NEURONS = 61
N_OUTPUTS = 1


def get_w(inputs, neurons, outputs):
    w_matrix = np.vstack(
        (  # Fully connected, generated randomly over interval
            np.hstack(
                (
                    np.random.uniform(0, 0.2, (inputs, neurons - outputs)),
                    np.zeros((inputs, outputs)),
                )
            ),
            np.hstack(
                (
                    np.zeros((neurons - outputs, neurons - outputs)),
                    np.random.uniform(0, 0.2, (neurons - outputs, outputs)),
                )
            ),
            np.zeros((outputs, neurons)),
        )
    )
    w_matrix = np.ma.array(np.float16(w_matrix), mask=(w_matrix == 0))

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
        "processing_time": 500,  # v/ 500ms
        "firing_threshold": 16,  # v/
        "magnitude": 1,  # v/
        "potential_decay": 0.05,  # v/ Decay constant Tau=20ms, lambda=e^(-t/T)
        "prob_rand_fire": 0.15,  # Seemingly 0 in paper but this is critical to learning.
        "refractory_period": 0,  # v/ Gutig, Aharonov, Rotter, & Sompolinsky 2003
        "learning_rate": 0.625 / 25,  # v/ gamma_0 = gamma / Tau_z
        "max_weight": 5,  # v/
        "stdp_window": 20,  # v/ Tau_+ = Tau_- = 20ms
        "trace_decay": 0.04,  # v/ T_z = 25, lambda = e^(-1/T_z)
        "action_threshold": 0,  # v/ Irrelevant
        "expected_value": expected_value,
        "continuous_rwd_action": continuous_rwd_action,
        "state_rate_map": [0, 0.08],  # v/ 40hz = 40spikes/500ms
        "punish_mult": 1,
    }
    parts = {
        "inputs": input.RateMap,  # Poisson
        "neurons": neuron.Neuron,  # v/
        "synapses": synapse.RLSTDPET,  # v/
        "weights": weight.Manual,  # v/
        "readout": readout.Threshold,  # v/
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
