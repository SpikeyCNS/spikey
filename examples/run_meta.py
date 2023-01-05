"""
Evolve a neural network to learn an RL enviornment.

https://docs.ray.io/en/latest/tune/index.html
"""
import numpy as np
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch


# 0. Define model and game
from spikey.snn import *
from spikey.games import Logic

class FlorianReward(spikey.snn.reward.template.Reward):
    def __call__(self, state, action, state_next):
        if sum(state) % 2 == 1:  # (0, 1) and (1, 0)
            return self._reward_mult if action == True else 0
        else:  # (0, 0) and (1, 1)
            return -self._punish_mult if action == True else 0

N_INPUTS = 60
N_NEURONS = 61
N_OUTPUTS = 1
N_HIDDEN = N_NEURONS - N_OUTPUTS
PROCESSING_TIME = 500

w_matrix = [
    np.random.uniform(0, .2, (N_INPUTS, N_HIDDEN)),
    np.random.uniform(0, .2, (N_HIDDEN, N_OUTPUTS)),
]

LOW_RATE = 0
HIGH_RATE = 40 / PROCESSING_TIME
state_rate_map = {# 2 input groups. 0hz when group false, 40hz when true
    (0, 0): np.array([LOW_RATE, LOW_RATE]),
    (0, 1): np.array([LOW_RATE, HIGH_RATE]),
    (1, 0): np.array([HIGH_RATE, LOW_RATE]),
    (1, 1): np.array([HIGH_RATE, HIGH_RATE]),
}

class network_template(ActiveRLNetwork):
    parts = {
        'inputs': input.RateMap,
        'neurons': neuron.Neuron,
        'synapses': synapse.RLSTDP,
        'weights': weight.Manual,
        'readout': readout.Threshold,
        'rewarder': FlorianReward,
    }
    keys = {
        "n_inputs": N_INPUTS,
        'n_neurons': N_NEURONS,
        "n_outputs": N_OUTPUTS,
        'matrix': w_matrix,

        'input_pct_inhibitory': .5,
        'neuron_pct_inhibitory': 0,
        'magnitude': 1,
        'firing_threshold': 16,
        'refractory_period': 0,  # Gutig, Aharonov, Rotter, & Sompolinsky 2003
        'prob_rand_fire': .15,
        'potential_decay': .05,  # Decay constant Tau=20ms, lambda=e^(-t/T)
        'trace_decay': .04,  # T_z = 25, lambda = e^(-1/T_z)
        "punish_mult": 1,

        'processing_time': PROCESSING_TIME,
        'learning_rate': .625 / 25,  # gamma_0 = gamma / Tau_z
        'max_weight': 5,
        'stdp_window': 20,  # Tau_+ = Tau_- = 20ms
        'action_threshold': 0,  # Makes network always output True

        'continuous_rwd_action': lambda network, game: network.spike_log[-1, -1],
        'state_rate_map': state_rate_map,
    }


# 1. Wrap a model in an objective function.
def objective(config):
    network_template.keys.update(config)
    game = Logic(preset="XOR")
    model = network_template()

    while True:  # Tune will cap it to training_iteration defined below
        model.reset()
        state = game.reset()
        state_next = None
        reward = 0

        for s in range(100):
            action = model.tick(state)
            state_next, _, done, __ = game.step(action)
            _ = model.reward(state, action, state_next)

            # Reward in florian = number of spikes in positive state, opposite of number spikes in negative state.
            if sum(state) % 2 == 1:  # (0, 1) and (1, 0)
                reward += model.rewarder._reward_mult * model.spike_log[-1, -1]
            else:  # (0, 0) and (1, 1)
                reward += -model.rewarder._punish_mult * model.spike_log[-1, -1]


            state = state_next
            if done:
                break

        session.report({"epoch_reward": reward})  # Report to Tune


# 2. Define a search space and initialize the search algorithm.
search_space = {
    "input_pct_inhibitory": tune.choice(list(np.arange(0, 1, 0.05))),
    "neuron_pct_inhibitory": tune.choice(list(np.arange(0, 1.0, 0.05))),
    "firing_threshold": tune.choice(list(range(1, 31))),
    "potential_decay": tune.choice(list(np.arange(0, 1, 0.02))),
    "trace_decay": tune.choice(list(np.arange(0, 1, 0.02))),
    "refractory_period": tune.choice(list(range(15))),
    "max_weight": tune.choice(list(np.arange(1, 10.1, 0.5))),
    "stdp_window": tune.choice(list(range(5, 100, 5))),
    "learning_rate": tune.choice([x / 25 for x in np.arange(0.01, 1.0, 0.01)]),
    "magnitude": tune.choice(list(np.arange(-10, 10.1, 0.5))),
    "reward_mult": tune.choice(list(np.arange(0, 5.1, 0.5))),
    "punish_mult": tune.choice(list(np.arange(0, 5.1, 0.5))),
}
algo = OptunaSearch()

# 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="epoch_reward",
        mode="max",
        search_alg=algo,
    ),
    run_config=air.RunConfig(
        stop={"training_iteration": 5},
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
