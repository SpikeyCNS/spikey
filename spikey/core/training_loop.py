"""
A custom training loop.
"""
from copy import deepcopy

from spikey.core.callback import RLCallback


class TrainingLoop:
    """
    A training loop template.

    Parameters
    ----------
    network_template: SNN[class]
        Network type to train.
    game_template: RL[class]
        Game type to train.
    params: dict
        Network, game and training parameters.
    """

    NECESSARY_CONFIG = {
        "n_episodes": "int Number of episodes to run,",
        "len_episode": "int Length of episode.",
    }

    def __init__(self, network_template: object, game_template: object, params: dict):
        self.network_template = network_template
        self.game_template = game_template

        self.params = deepcopy(self.network_template.config)
        self.params.update(params)

    def reset(
        self,
        network_template: object = None,
        game_template: object = None,
        params: dict = None,
    ) -> (object, object):
        """
        Reset.
        """
        if network_template is not None:
            self.network_template = network_template
        if game_template is not None:
            self.game_template = game_template
        if params is not None:
            params.update(params)

        game = self.game_template(**self.params)
        network = self.network_template(callback=experiment, game=game, **self.params)

        return network, game

    def __call__(self, **kwargs) -> object:
        """
        Execute training loop.
        """
        raise NotImplementedError(f"Call not implemented in {type(self)}.")


class GenericLoop(TrainingLoop):
    def __call__(self, **kwargs) -> ("SNN", "RL", dict, dict):
        if "callback" in kwargs:
            experiment = kwargs["callback"]
        else:
            experiment = RLCallback(
                **self.params,
                reduced=kwargs["reduced"] if "reduced" in kwargs else False,
            )

        experiment.reset()
        game = self.game_template(callback=experiment, **self.params)
        network = self.network_template(callback=experiment, game=game, **self.params)

        for e in range(self.params["n_episodes"]):
            network.reset()
            state = game.reset()

            for s in range(self.params["len_episode"]):
                action = network.tick(state)

                state, _, done, __ = game.step(action)

                reward = network.reward(state, action)

                if done:
                    break

        experiment.training_end()

        return [*experiment]


class TDLoop(TrainingLoop):
    def __call__(self, **kwargs) -> ("SNN", "RL", dict, dict):
        if "callback" in kwargs:
            experiment = kwargs["callback"]
        else:
            experiment = RLCallback(
                **self.params,
                reduced=kwargs["reduced"] if "reduced" in kwargs else False,
            )
        experiment.track(
            "network_reward",
            "info",
            "td_td",
            ["network", "rewarder", "prev_td"],
            "list",
        )
        experiment.track(
            "network_reward",
            "info",
            "td_reward",
            ["network", "rewarder", "prev_reward"],
            "list",
        )
        experiment.track(
            "network_reward",
            "info",
            "td_value",
            ["network", "rewarder", "prev_value"],
            "list",
        )
        experiment.reset()

        game = self.game_template(callback=experiment, **self.params)
        network = self.network_template(callback=experiment, game=game, **self.params)

        try:
            for e in range(self.params["n_episodes"]):
                network.reset()
                state = game.reset()

                network.rewarder.prev_value = None
                network.rewarder.time = 0

                for s in range(self.params["len_episode"]):
                    action = network.tick(state)

                    state, _, done, __ = game.step(action)

                    if network.readout._n_outputs:
                        network.rewarder.critic_spikes = network.spike_log[
                            -network._processing_time :,
                            -network._n_neurons : -network.readout._n_outputs,
                        ]
                    else:
                        network.rewarder.critic_spikes = network.spike_log[
                            -network._processing_time :, -network._n_neurons :
                        ]

                    reward = network.reward(state, action)

                    if done:
                        break

        except KeyboardInterrupt:
            pass

        experiment.training_end()

        return [*experiment]
