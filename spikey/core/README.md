# Core Experiment Tools

These Core experiment tools are what connect the pieces of the Spikey together in order to run experiments.
TrainingLoop is the centerpiece for experiments within Spikey, it allows users to train a network on a game a single time, repeatedly or within the meta analysis tools.
ExperimentCallback is the core of Spikey's experiment management, it gathers various network, game and experiment signals during training.
The Callback's results can be used directly by the logging and viz tools.

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

## Monitoring Signals

It is best practice to use the Callback object whenever you seek to monitor some signal within Spikey.
This will make it easy to monitor the same signal regardless of training
loop and simplify the sharing of this data to logging and viz tools.

A callback can be optionally passed to, and thus shared between both a game
and network, shared automatically if using the pre-built TrainingLoops.
Every time one of the network or game's methods are executed, the callback
will be alerted via a function of similar name, eg you call ```network.tick(state)``` which calls ```callback.network_tick(*inputs, *outputs)```.
The parameters of this callback function are the inputs and output of the
original method.
Each callback has two important member dictionaries, results and info. Results are for storing scalar variables that can be easily loaded into a table, info may contain ndarrays and arbitrary (serializable) objects.
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
