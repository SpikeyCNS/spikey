from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from spikey.core import *
from spikey.snn import *
from spikey.RL import *

np.random.seed(0)



def print_rates(experiment_output, training_params):
    _, __, ___, info = experiment_output

    states = np.array(info['step_states'])
    inrates = np.array(info['step_inrates'])
    outrates = np.array(info['step_outrates'])

    for state in range(10):
        mean_inrates = np.mean(inrates[states == state])
        mean_outrates = np.mean(outrates[states == state])

        print(f"{state}: {mean_inrates:.4f} -> {mean_outrates:.4f}")

N_STATES = 10


class game_template(Logic):
    def _get_state(self) -> np.ndarray:
        return np.random.randint(N_STATES)


training_params = {
    'n_episodes': 10,
    'len_episode': 100,
}

N_INPUTS = 100
N_NEURONS = 50
N_OUTPUTS = N_NEURONS

FIRE_STATES = [0, 3, 6, 9]  # States network should fire in

w_matrix = np.vstack((  # Feedforward, single layer
    np.random.uniform(0, .5, (N_INPUTS, N_NEURONS)),
    np.zeros((N_NEURONS, N_NEURONS)),
))
w_matrix = np.ma.array(np.float16(w_matrix), mask=(w_matrix == 0), fill_value=0)

state_rate_map = np.zeros((N_STATES, N_STATES), dtype=np.float)
for state in range(N_STATES):#FIRE_STATES:
    state_rate_map[state, state] = .2

class network_template(RLNetwork):
    config = {
        "n_inputs": N_INPUTS,
        'n_neurons': N_NEURONS,
        "n_outputs": N_OUTPUTS,
        'matrix': w_matrix,
        'magnitude': 1,
        'potential_decay': .05,
        'output_range': [0, 1],
        'trace_decay': .04,

        'input_pct_inhibitory': 0,
        'neuron_pct_inhibitory': 0,
        'prob_rand_fire': 0,
        'refractory_period': 0,
        'firing_threshold': 10,

        'processing_time': 100,
        'learning_rate': .01,
        'max_weight': 5,
        'stdp_window': 20,

        'reward_mult': 1,
        'punish_mult': 0,
        'action_threshold': .010,  # Most critical, no reward recieved unless surpasses this threshold

        'expected_value': lambda state: state in FIRE_STATES,
        'state_rate_map': state_rate_map, 
    }
    _template_parts = {
        'inputs': input.RateMap,
        'neurons': neuron.Neuron,
        'synapses': synapse.RLSTDPET,
        'weights': weight.Manual,
        'readout': readout.Threshold,
        'rewarder': reward.MatchExpected,
        'modifiers': None,
    }

# Control, without learning
callback = RLCallback(**training_params, reduced=False, measure_rates=True)
training_loop = GenericLoop(network_template, game_template, training_params)
training_loop.reset(params={'learning_rate': 0, 'n_episodes': 2})
e_output = training_loop(callback=callback)

print(f"{callback.results['total_time']:.2f}s")
print_rates(e_output, training_params)

print('\n\n\n')
# Real test
callback = RLCallback(**training_params, reduced=False, measure_rates=True)
training_loop = GenericLoop(network_template, game_template, training_params)
e_output = training_loop(callback=callback)

print(f"{callback.results['total_time']:.2f}s")
print_rates(e_output, training_params)
