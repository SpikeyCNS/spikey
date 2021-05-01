# Core Experiment Tools

Core experiment tools are what connect the pieces of the
Spikey together. 
TrainingLoop allows users to train a network on a game individually, repeatedly or within the meta analysis tools. TrainingLoop is the centerpiece for experiments within spikey.
ExperimentCallback gathers various network, game and general experiment data while training. Callback is the core of Spikey's experiment management and its format is used by the logger and reader.

```none
--------   -------------
| Game |   |  Network  |
--------   -------------    ------------
   |____________/___________| Callback | --> Logger --> Reader
   |           /            ------------
-----------------
| Training Loop |
-----------------
```

The ability to remember and understand experiment results is crucial for
making progress towards a goal.
This task becomes difficult with spiking neural networks given their stochastic nature and inherent complexity.
Often it is important to run the same experiment multiple times in order to gauge algorithm effectiveness, a hyperparameter search which produces much data may be necessary to solve a problem or sometimes a combination of both is needed.
Out of the box, this package provides data monitoring, analyzing, logging and the corresponding log reading tools.
Each group of tools contains functionality to analyze a single or an aggregate of experiments.

## Monitoring Signals

Although it is possible to custom write code to monitor some variable,
it is best practice to use a callback object.
This will make it easy to monitor the same signal regardless of training
loop and simplify the sharing of this data across the platform.

A callback can be optionally passed to, and thus shared between both a game
and network.
Every time one of the network or game's methods are executed, the callback
will be alerted via a function of similar name, eg ```callback.network_tick(*inputs, *outputs)```.
The parameters of this callback function are the inputs and output of the
original method.
Each callback has two important member dictionaries, results and info. Results are for storing scalar variables that can be easily loaded into a table, info may contain ndarrays and generic(serializable) objects.
When done training, ```callback.log()``` can be used to generate
a log file with all network and game parameters as well as the contents
of results and info.

The user may start with a blank slate, ExperimentCallback, and define any
or all network and game functions, otherwise they may override and extend
a decent baseline eg, RLCallback.
At runtime, either of these options may be extended via ```callback.monitor("<network/game>_<methodname>", "<results/info>", "<key>", target=["<network/game>", "<part_name>", "<variable_name>"], method="<list/scalar>")```.

[Callback implementations here](https://github.com/SpikeyCNS/spikey/blob/master/spikey/core/callback.py).

```python
"""
Signal monitoring demonstration, creating ndarrays with TD reward data.
"""
import spikey

callback = spikey.core.RLCallback()
callback.monitor(
    "network_reward",
    "info",
    "td_td",
    ["network", "rewarder", "prev_td"],
    "list"
)
callback.monitor(
    "network_reward",
    "info",
    "td_reward",
    ["network", "rewarder", "prev_reward"],
    "list",
)
callback.monitor(
    "network_reward",
    "info",
    "td_value",
    ["network", "rewarder", "prev_value"],
    "list",
)

training_loop = spikey.core.GenericLoop(network_template, game_template, callback, training_params)
network, game, results, info = training_loop()
```
