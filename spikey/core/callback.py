"""
Callbacks to tie into network & game.
"""
import os
from time import time

import numpy as np

from spikey.logging import log


def get_spikey_version() -> str:
    """
    Find version of spikey being used.

    Returns
    -------
    str Spikey version.
    """
    #import spikey
    try:  # Python >= 3.8
        from importlib.metadata import version
        return version("spikey")
    except ImportError:
        pass

    try:  # Python < 3.8
        import pkg_resources
        return pkg_resources.get_distribution("spikey").version
    except ImportError:
        pass

    try:  # Dev version
        from setup import setup_args
        return setup_args['version']
    except ImportError:
        pass

    print("[WARNING] Failed to find spikey version!")
    return "UNDEFINED"


class ExperimentCallback:
    """
    Track experiment data.

    Parameters
    ----------
    kwargs: dict
        Training parameters.
    """

    NECESSARY_CONFIG = {
        "n_episodes": "int Number of episodes to run,",
        "len_episode": "int Length of episode.",
    }

    def __init__(self, **kwargs):
        self.experiment_params = kwargs

        self.network, self.game = None, None
        self.results, self.info = None, None

        self.tracking = {}
        self.wrap_all()

    def __enter__(self) -> "self":
        return self

    def __exit__(self, *args):
        self.training_end()

    def __getattr__(self, key: str) -> callable:
        """
        Called when {key} hasn't already been defined.
        """
        return lambda *a, **kw: False

    def __iter__(self):
        yield self.network
        yield self.game
        yield self.results
        yield self.info

    def reset(self):    
        self.results, self.info = {}, {}

        try:
            self.results["version"] = get_spikey_version()
        except Exception as e:
            print(f"Failed to find spikey version! '{e}'")
            self.results["version"] = None

        for _, value in self.tracking.items():
            for location, identifier, __, method in value:
                self.__dict__[location][identifier] = (
                    [] if method == "list" else 0
                )

    def track_wrapper(self, func: callable, funcname: str) -> callable:
        def track_wrap(*args, **kwargs):
            output = func(*args, **kwargs)

            if funcname == 'network_reset':
                for name, values in self.tracking.items():
                    for location, identifier, target, method in values:
                        if method == "list":
                            self.__dict__[location][identifier].append([])
 
            if funcname in self.tracking:
                for location, identifier, target, method in self.tracking[funcname]:
                    try:
                        if callable(target):
                            item = target()
                        else:
                            item = self
                            for name in target:
                                if name == "()":
                                    item = item()
                                elif name == "arg":
                                    item = kwargs
                                elif name.startswith("arg"):
                                    _, idx = name.split('_')
                                    item = args[int(idx)]
                                elif isinstance(item, (dict, list, tuple)):
                                    item = item[name]
                                else:
                                    item = getattr(item, name)
                    except Exception as e:
                        print(f"Failed to get {'.'.join(target)} in {type(self)}: {e}.")

                    if isinstance(item, np.ndarray):
                        item = np.copy(item)

                    if method == "list":
                        self.__dict__[location][identifier][-1].append(item)
                    elif method == "scalar":
                        self.__dict__[location][identifier] = item
                    else:
                        raise ValueError(f"Unrecognized method {method}!")

            return output

        return track_wrap

    def wrap_all(self):
        """
        Wrap all function in class.
        """
        for key in dir(self):
            value = getattr(self, key)

            if (
                not callable(value)
                or key[0] == "_"
                or key in ["track_wrapper", "wrap_all"]
            ):
                continue

            setattr(self, key, self.track_wrapper(value, key))

    def track(
        self,
        function: str,
        location: str,
        identifier: str,
        target: list,
        method: str = "list",
    ):
        """
        Start tracking new information.

        Parameters
        ----------
        function: str
            Name of callback to attach to.
        location: str
            Storage location, 'results' or 'info'.
        identifier: str
            Key to save as
        target: list[str]
            Location of information, eg ['network', 'synapse', 'spike_log'].
            arg, arg_<int> are reserved for accessing kwargs and list[<int>] respectively.
        method: 'scalar' or 'list'
            Whether to store as list or scalar.
        """
        if function not in self.tracking:
            self.tracking[function] = []

        try:
            self.tracking[function].append((location, identifier, target, method))
        except KeyError:
            raise KeyError(f"Failed to find {function} in {type(self)}.")

    def log(self, **kwargs):
        """
        kwargs = folder or file
        """
        log(self.network, self.game, self.results, self.info, **kwargs)


class RLCallback(ExperimentCallback):
    def __init__(self, reduced: bool = False, measure_rates: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.reduced = reduced
        self._measure_rates = measure_rates

        self.track('network_init', 'info', 'start_time', time, 'scalar')
        self.track('network_tick', 'info', 'step_states', ['arg_0'], 'list')
        self.track('network_tick', 'info', 'step_actions', ['arg_1'], 'list')
        self.track('network_reward', 'info', 'step_rewards', ['arg_2'], 'list')
        self.track('training_end', 'info', 'finish_time', time, 'scalar')
        self.track('training_end', 'results', 'total_time', lambda: self.info["finish_time"] - self.info["start_time"], 'scalar')
        self.track('training_end', 'results', 'n_episodes', ['experiment_params', 'n_episodes'], 'scalar')
        self.track('training_end', 'results', 'len_episode', ['experiment_params', 'len_episode'], 'scalar')
        if not self.reduced:
            self.track('network_init', 'info', 'weights_original', ['network', 'synapses', 'weights', 'matrix'], 'scalar')
            self.track('training_end', 'info', 'weights_final', ['network', 'synapses', 'weights', 'matrix'], 'scalar')

    def reset(self):
        super().reset()

        self.info["episode_lens"] = []

        if self._measure_rates:
            (
                self.info["step_inrates"],
                self.info["step_sysrates"],
                self.info["step_outrates"],
            ) = ([], [], [])

    def network_init(self, network: "SNN"):
        self.network = network

    def game_init(self, game: "RL"):
        self.game = game

    def network_reset(self):
        if self._measure_rates:
            self.info["step_inrates"].append([])
            self.info["step_sysrates"].append([])
            self.info["step_outrates"].append([])

        self.info["episode_lens"].append(0)

    def game_reset(self, state: object):
        pass

    def network_tick(self, state: object, action: object):
        if self._measure_rates:
            relevant_spikes = np.abs(
                self.network.spike_log[
                    -self.network._processing_time :, -self.network._n_outputs :
                ]
            )

            outrate = np.mean(relevant_spikes)

            inrate = np.mean(
                np.abs(
                    self.network.spike_log[
                        -self.network._processing_time :, : self.network._n_inputs
                    ]
                )
            )
            sysrate = np.mean(
                np.abs(
                    self.network.spike_log[
                        -self.network._processing_time :,
                        self.network._n_inputs : -self.network._n_outputs,
                    ]
                )
            )

            self.info["step_inrates"][-1].append(inrate)
            self.info["step_sysrates"][-1].append(sysrate)
            self.info["step_outrates"][-1].append(outrate)

        self.info["episode_lens"][-1] += 1

    def game_step(
        self,
        action: object,
        state: object,
        state_new: object,
        rwd: float,
        done: bool,
        info: dict,
    ):
        pass

    def network_reward(self, state: object, action: object, reward: float):
        pass

    def training_end(self):
        pass
