import numpy as np

from spikey.snn import *
from spikey.games import Logic
from spikey.viz import print_rates

from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig, RunConfig

np.random.seed(0)


class Network(spikey.Network):
    def __init__(self):
        N_INPUTS = 60
        N_NEURONS = 61
        N_OUTPUTS = 1
        N_HIDDEN = N_NEURONS - N_OUTPUTS


        w_matrix = np.zeros((N_INPUTS + N_NEURONS, N_INPUTS + N_NEURONS))
        w_matrix[:N_INPUTS, N_INPUTS:-N_OUTPUTS] = np.random.uniform(0, .2, (N_INPUTS, N_HIDDEN))
        w_matrix[N_INPUTS:-N_OUTPUTS, -N_OUTPUTS:] = np.random.uniform(0, .2, (N_HIDDEN, N_OUTPUTS))
        self.synapse = synapse.RLSTDP(w_matrix, n_inputs=N_INPUTS, n_neurons=N_NEURONS, trace_decay=.04, learning_rate=.625 / 25, max_weight=5, stdp_window=20)

        LOW_RATE = 0
        HIGH_RATE = 40 / 500
        state_rate_map = {# 2 input groups. 0hz when group false, 40hz when true
            (0, 0): np.array([LOW_RATE, LOW_RATE]),
            (0, 1): np.array([LOW_RATE, HIGH_RATE]),
            (1, 0): np.array([HIGH_RATE, LOW_RATE]),
            (1, 1): np.array([HIGH_RATE, HIGH_RATE]),
        }
        self.input = input.RateMap(n_inputs=N_INPUTS, n_neurons=N_NEURONS, magnitude=1, state_rate_map=state_rate_map, input_pct_inhibitory=.5)

        self.n1 = neuron.Neuron(n_inputs=N_INPUTS, n_neurons=N_NEURONS, neuron_pct_inhibitory=0., magnitude=1, firing_threshold=16, refractory_period=0, prob_rand_fire=.15, potential_decay=.05)
        self.n2 = neuron.Neuron(n_inputs=N_NEURONS, n_neurons=N_OUTPUTS, neuron_pct_inhibitory=0., magnitude=1, firing_threshold=16, refractory_period=0, prob_rand_fire=.15, potential_decay=.05)
        self.readout = readout.Threshold(n_outputs=N_OUTPUTS, magnitude=1, action_threshold=0, continuous_rwd_action=lambda network, game: network.spike_log[-1, -1],)

    def reset(self):
        self.synapse.reset()
        self.input.reset()
        self.n1.reset()
        self.n2.reset()
        self.readout.reset()

    def tick(self, state):
        self.input.update(state)
        spike_log = np.zeros((500, self.input._n_inputs + self.n1._n_neurons))
        for i in range(500):
            # TODO lost timing info on this
            x = np.zeros(self.input._n_inputs + self.n1._n_neurons)
            x[:self.input._n_inputs] = self.input()
            x = self.synapse * x
            x[self.input._n_inputs:] = self.n1(x[self.input._n_inputs:])
            x = self.synapse * x
            x[-self.n2._n_neurons:] = self.n2(x[-self.n2._n_neurons:])
            micro_action = x[-1]  # self.readout(x)
            yield micro_action
            spike_log[i] = x
        self.synapse.update(spike_log, np.ones(self.input._n_inputs + self.n1._n_neurons))

    def reward(self, state, action, state_new, game_rwd):
        if sum(state) % 2 == 1:  # (0, 1) and (1, 0)
            rwd = 1 if action == True else 0
        else:  # (0, 0) and (1, 1)
            rwd = -1 if action == True else 0
        self.synapse.reward(rwd)
        return rwd


if __name__ == '__main__':
    game = Logic(preset="XOR")
    model = Network()

    for epoch in range(800):
        model.reset()
        state = game.reset()
        state_next = None

        total_reward = 0
        for s in range(1):
            for micro_action in model.tick(state):
                reward = model.reward(state, micro_action, state_next, None)
                total_reward += reward
            state_next, _, done, __ = game.step(micro_action)
            state = state_next
            if done:
                break

        print(total_reward)
