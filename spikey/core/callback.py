"""
Implementations of experiment callbacks for tracking network and
game parameters during experiment runs.
"""
import os
from time import time
from copy import copy, deepcopy

import numpy as np

from spikey.logging import log


def get_spikey_version() -> str:
    """
    Get version of spikey import.

    Returns
    -------
    str Spikey version.
    """
    # import spikey
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

        return setup_args["version"]
    except ImportError:
        pass

    print("[WARNING] Failed to find spikey version!")
    return "UNDEFINED"


class ExperimentCallback:
    """
    Base experiment callback for tracking network and game parameters
    during experiment runs.

    If you would like to add callback support to a new network
    or game method, simply add,
    ```python
    self.callback.<tracking_identifier>(*method_params, *method_returns)
    ```
    to the end of the method. Tracking identifier can be `game_tick`,
    `network_reward` or any unique identifier. Make use of this identifier
    either by defining a method of the same name within the callback or by
    using a runtime tracker(see Runtime Tracking below).

    Parameters
    ----------
    **kwargs: dict
        For compat with TrainingLoop.

    Usage
    -----
    ```python
    callback = ExperimentCallback()
    callback.reset()

    # Run training loop

    callback.log(filename='output.json')
    ```

    Runtime Tracking
    ----------------
    ```python
    callback = ExperimentCallback()
    callback.track('network_tick', 'info', 'step_actions', ['arg_1'], 'list')
    callback.reset()
    ```
    """

    def __init__(self, **kwargs):
        self.network, self.game = None, None
        self.results, self.info = None, None

        self.tracking = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.training_end()

    def __deepcopy__(self, memo):
        """
        Return a deepcopy of self.
        """
        cls = self.__class__
        callback = cls.__new__(cls)
        memo[id(self)] = callback
        for k, v in self.__dict__.items():
            setattr(callback, k, deepcopy(v, memo))
        return callback

    def __getstate__(self):
        # TODO should probably add helper to warn against lambdas
        return {key: getattr(self, key) for key in ['network', 'game', 'results', 'info', 'tracking']}

    def __setstate__(self, items):
        self.__dict__.update(items)

    def __getattr__(self, key: str) -> callable:
        """
        Called when {key} hasn't already been defined.
        """
        return lambda *a, **kw: False

    def __iter__(self):
        """
        For trainingloop, makes `return *callback` == `return network, game, results, info`.
        """
        yield self.network
        yield self.game
        yield deepcopy(self.results)
        yield deepcopy(self.info)

    def _extend_list_trackers(self):
        """
        Extend list trackers to new episode,
        eg [[1, 2, 3]] -> [[1, 2, 3], []]
        """
        for name, values in self.tracking.items():
            for location, identifier, target, method in values:
                if method == "list":
                    self.__dict__[location][identifier].append([])

    def _track_wrapper(self, func: callable, funcname: str) -> callable:
        """
        Wrap function in callback for tracking behavior.
        """

        def track_wrap(*args, **kwargs):
            output = func(*args, **kwargs)

            if funcname == "network_reset":
                self._extend_list_trackers()

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
                                    _, idx = name.split("_")
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
                        dest = self.__dict__[location][identifier]
                        if len(dest) and isinstance(dest[0], list):
                            dest[-1].append(item)
                        else:
                            # TODO potentially raise warning that its nonstandard usage
                            # for callback before network_reset
                            dest.append(item)
                    elif method == "scalar":
                        self.__dict__[location][identifier] = item
                    else:
                        raise ValueError(f"Unrecognized method {method}!")

            return output

        return track_wrap

    def _wrap_all(self):
        """
        Wrap all functions in callback for runtime tracking behavior.
        """
        for key in dir(self):
            value = getattr(self, key)

            if (
                not hasattr(value, "__name__")
                or value.__name__ == "track_wrap"  # has been wrapped
                or not callable(value)
                or key[0] == "_"
                or key in ["_track_wrapper", "_wrap_all"]
            ):
                continue

            setattr(self, key, self._track_wrapper(value, key))

    def reset(self, **experiment_params):
        """
        Reset callback, overwrites all previously collected data.

        Parameters
        ----------
        **experiment_params: dict, default=None
            Used like kwargs eg, `RLCallback(**experiment_params)`.
            Experiment setup parameters(not network & game params).

        Usage
        -----
        ```python
        callback.reset()
        ```
        """
        self._wrap_all()

        self.results, self.info = {}, {}

        try:
            self.results["version"] = get_spikey_version()
        except Exception as e:
            print(f"Failed to find spikey version! '{e}'")
            self.results["version"] = None

        self.results.update(experiment_params)

        for _, value in self.tracking.items():
            for location, identifier, __, method in value:
                self.__dict__[location][identifier] = [] if method == "list" else 0

    def bind(self, name):
        """
        Add binding for trackers and later referencing.
        All bindings must be set before calling reset.
        """
        setattr(self, name, lambda *a, **kw: None)

    def track(
        self,
        function: str,
        location: str,
        identifier: str,
        target: list,
        method: str = "list",
    ):
        """
        Setup runtime tracking for a new parameter.

        Parameters
        ----------
        function: str
            Name of callback to attach to.
        location: str
            Storage location, 'results' or 'info'.
        identifier: str
            Key to save target in.
        target: list[str]
            Location of information, eg ['network', 'synapse', 'spike_log'].
            arg, arg_<int> are reserved for accessing kwargs and list[<int>] respectively.
        method: 'scalar' or 'list'
            Tracking method, whether to store as list or scalar.

        Usage
        -----
        ```python
        callback.track('training_end', 'results', 'processing_time', ['network', 'processing_time'], 'scalar')
        ```

        ```python
        callback.track('network_tick', 'info', 'step_actions', ['arg_1'], 'list')
        ```
        """
        if function not in self.tracking:
            self.tracking[function] = []

        try:
            self.tracking[function].append((location, identifier, target, method))
        except KeyError:
            raise KeyError(f"Failed to find {function} in {type(self)}.")

    def log(self, log_func=log, **log_kwargs):
        """
        Log data to file.

        Parameters
        ----------
        log_kwargs: dict
            Log function kwargs
        """
        log_func(self.network, self.game, self.results, self.info, **log_kwargs)


class RLCallback(ExperimentCallback):
    """
    Experiment callback for tracking network and game parameters
    during reinforcement learning experiment runs.

    If you would like to add callback support to a new network
    or game method, simply add,
    ```python
    self.callback.<tracking_identifier>(*method_params, *method_returns)
    ```
    to the end of the method. Tracking identifier can be `game_tick`,
    `network_reward` or any unique identifier. Make use of this identifier
    either by defining a method of the same name within the callback or by
    using a runtime tracker(see Runtime Tracking below).

    Parameters
    ----------
    reduced: bool, default=False
        Reduced amount of logging or not.
    measure_rates: bool, default=False
        Whether or not to measure network input, body and output rates at
        each step - is time consuming.
    **kwargs: dict
        For compat with TrainingLoop.

    Usage
    -----
    ```python
    callback = RLCallback()
    callback.reset()

    # Run training loop

    callback.log(filename='output.json')
    ```

    Runtime Tracking
    ---------------
    ```python
    callback = RLCallback()
    callback.track('network_tick', 'info', 'step_actions', ['arg_1'], 'list')
    callback.reset()
    ```
    """

    def __init__(self, reduced: bool = False, measure_rates: bool = False, **kwargs):
        super().__init__()

        self.reduced = reduced
        self._measure_rates = measure_rates

        self.track("network_init", "info", "start_time", time, "scalar")
        self.track("network_tick", "info", "step_states", ["arg_0"], "list")
        self.track("network_tick", "info", "step_actions", ["arg_1"], "list")
        self.track("network_reward", "info", "step_rewards", ["arg_2"], "list")
        if not self.reduced:
            self.track(
                "network_continuous_reward", "info", "tick_rewards", ["arg_2"], "list"
            )
        self.track("training_end", "info", "finish_time", time, "scalar")
        self.track(
            "training_end",
            "results",
            "total_time",
            self._total_time,
            "scalar",
        )
        if not self.reduced:
            self.track(
                "network_init",
                "info",
                "weights_original",
                ["network", "synapses", "weights", "matrix"],
                "scalar",
            )
            self.track(
                "training_end",
                "info",
                "weights_final",
                ["network", "synapses", "weights", "matrix"],
                "scalar",
            )

    def _total_time(self):
        return self.info["finish_time"] - self.info["start_time"]

    def reset(self, **experiment_params):
        """
        Reset callback, overwrites all previously collected data.

        Parameters
        ----------
        **experiment_params: dict, default=None
            Used like kwargs eg, `RLCallback(**experiment_params)`.
            Experiment setup parameters(not network & game params).

        Usage
        -----
        ```python
        callback.reset()
        ```
        """
        super().reset(**experiment_params)

        self.info["episode_lens"] = []

        if self._measure_rates:
            (
                self.info["step_inrates"],
                self.info["step_sysrates"],
                self.info["step_outrates"],
            ) = ([], [], [])

    def network_init(self, network: object):
        self.network = network

    def game_init(self, game: object):
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

    def network_continuous_reward(self, state: object, action: object, reward: float):
        pass

    def training_end(self):
        pass


class TDCallback(RLCallback):
    """
    Experiment callback for tracking network and game parameters
    during reinforcement learning experiment runs. This callback
    builds on top of RLCallback plus a few common TD related
    trackers.

    If you would like to add callback support to a new network
    or game method, simply add,
    ```python
    self.callback.<tracking_identifier>(*method_params, *method_returns)
    ```
    to the end of the method. Tracking identifier can be `game_tick`,
    `network_reward` or any unique identifier. Make use of this identifier
    either by defining a method of the same name within the callback or by
    using a runtime tracker(see Runtime Tracking below).

    Usage
    -----
    ```python
    callback = TDCallback()
    callback.reset()

    # Run training loop

    callback.log(filename='output.json')
    ```

    Runtime Tracking
    ---------------
    ```python
    callback = TDCallback()
    callback.track('network_tick', 'info', 'step_actions', ['arg_1'], 'list')
    callback.reset()
    ```
    """

    def __init__(self, reduced: bool = False, measure_rates: bool = False):
        super().__init__()

        self.track(
            "network_continuous_reward",
            "info",
            "td_td",
            ["network", "rewarder", "prev_td"],
            "list",
        )
        self.track(
            "network_continuous_reward",
            "info",
            "td_reward",
            ["network", "rewarder", "prev_reward"],
            "list",
        )
        self.track(
            "network_continuous_reward",
            "info",
            "td_value",
            ["network", "rewarder", "prev_value"],
            "list",
        )
