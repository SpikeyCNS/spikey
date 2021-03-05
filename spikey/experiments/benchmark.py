"""
Baseline network, game and trainingloop setup.

Usage
-----
```python
from spikey.experiments.benchmark import (
    Loop,
    network_template,
    game_template,
    training_params,
)
```
"""
import numpy as np

from spikey.snn import *
from spikey.RL import *
from spikey.core import TrainingLoop, RLCallback


class Loop(TrainingLoop):
    def __call__(self, **kwargs):
        experiment = RLCallback(
            **self.params,
            reduced=kwargs["reduced"] if "reduced" in kwargs else False,
            measure_rates=True
        )
        experiment.reset()

        game = self.game_template(callback=experiment)
        network = self.network_template(game=game, **self.params)

        try:
            n_successes = 0
            for e in range(self.params["n_episodes"]):
                network.reset()
                state = game.reset()
                state_next = None

                for s in range(self.params["len_episode"]):
                    action = network.tick(state)

                    state_next, _, done, __ = game.step(action)

                    reward = network.reward(state, action)
                    state = state_next

                    if done:
                        break

        except KeyboardInterrupt:
            pass

        experiment.training_end()

        return network, game, experiment.results, experiment.info


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
            np.zeros((N_OUTPUTS, neurons)),
        )
    )
    w_matrix = np.ma.array(np.float16(w_matrix), mask=(w_matrix == 0))

    return w_matrix


class network_template(FlorianSNN):
    config = {
        "n_outputs": N_OUTPUTS,
        "matrix": get_w(N_INPUTS, N_NEURONS, N_OUTPUTS),  # v/
        "n_neurons": N_NEURONS,  # v/
        "input_pct_inhibitory": 0.5,  # v/
        "neuron_pct_inhibitory": 0,  # v/
        "processing_time": 500,  # v/ 500ms NOTENOTENOTENOTENOTENOTENOTENOTENOTENOTENOTE
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
        "state_rate_map": [0, 0.08],  # v/ 40hz = 40spikes/500ms
        "punish_mult": 1,
    }
    _template_parts = {
        "inputs": input.RateMap,  # Poisson
        "neurons": neuron.Neuron,  # v/
        "synapses": synapse.RLSTDPET,  # v/
        "weights": weight.Manual,  # v/
        "readout": readout.Threshold,  # v/
        "rewarder": reward.MatchExpected,
        "modifiers": None,  # TODO Learning rate small while accuracy high
    }


class game_template(Logic):
    config = Logic.PRESETS["XOR"]


training_params = {
    "n_episodes": 1,
    "len_episode": 800,
    "eval_steps": 50,
}

__ALL__ = [Loop, network_template, game_template, training_params]
