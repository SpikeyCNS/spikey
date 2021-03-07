# Spikey

Spikey is a malleable, [ndarray based](https://numpy.org/doc/stable/reference/arrays.ndarray.html) spiking neural network framework and training platform.

Contains many pre-made components, experiments and (meta)analysis tools.

Tested with Python 3.7, 3.8 in Linux and Windows. Expect to post bugs or suggestions in the issues tab :).

## Table of Contents

* [Spiking Neural Networks](#Spiking-Neural-Networks)
* [Package Overview](#Package-Overview)
* [Installation](#Installation)
  * [Local Installation](#Installation)
  * [Run Tests](https://github.com/SpikeyCNS/spikey#run-tests)
  * [Build Documentation](https://github.com/SpikeyCNS/spikey#build-documentation)
* [Getting Started](#Getting-Started)
* [Spikey Features](#Spikey-Features)
  * [Experiment Management](#Experiment-Management)
  * [Meta Analysis](#Meta-Analysis)
  * [Extending Functionality](#Extending-Functionality)
* [Contributing](#Contributing)

## Spiking Neural Networks

### What is a Spiking Neural Network?

Spiking neural networks are biologically plausible neural models that
handle temporal information well.
These are clusters of neurons interconnected by directed
synapses that aim to replicate(understand) the behavior of some system.

Spiking neurons are simple machines with an internal charge that slowly decays
over time, which only increase when another neuron fires through a synapse into it.
If this internal potential surpasses some firing threshold, the neuron will fire,
and its charge will reset and remain static for the duration of its refractory period.
This simple temporal behavior enables spiking neurons to communicate with eachother by means of
population, rate and fire order coding.
A special batch of the network's neurons ignore their membrane potentials,
and serve to translate sensory
information from the outside world into spike patterns that the rest of the group is
able to reason with; these are network inputs.
A separate subset of neurons is designated as the networks outputs.
These behave normally, but their spikes are interpreted by a readout function which dictates
the consensus of the crowd.
Information comes into and flows through the network encoded as spike patterns
in terms of firing rates, firing orders and even more complex patterns.
This natural ability to reason with chronological events makes spiking neural
networks an ideal candidate for modelling systems with temporal behaviors.

Learning in an artificial neural networks is largely facilitated by
an algorithm that tunes synapse weights.
The weight of a synapse between two neurons modulates how much current
goes from the pre- to the post-synaptic neuron, ie
```neuron_b.potential += neuron_a.output * synapse.weight```.
The learning algorithm used to tune the network must be able to handle
the temporal relationship between pre and post neuron fires, thus variants of the
hebbian rule are commonly used.
The hebbian rule acts on a per synapse basis, only considering the firing times
of the specific synapse's single pre- and post-synaptic neurons.
If the input neuron tends to fire before the output neuron, the algorithm will
increase the synaptic weight between the two.
Otherwise if the opposite pattern holds the weight will decrease.
In aggregate, the network learns detect patterns of the stimulus it is trained on.

A complex learning process emerges from these many simple interactions, with pattern detection
occuring at multiple scales.
This empowers spiking neural networks to understand games played over time!
Therefore these models are ideal candidates for environments modelled
as Markov decision processes.
Such games have been solved under the reinforcement learning
paradigm by modulating the STDP(pattern detection) updates with
a reinforcement signal in order to condidion network responses.
See Florian(2007) and Fremaux(2013) below for more on RLSTDP.

### Further Reading

* [Ponulak F, Kasinski A (2011) Introduction to spiking neural networks: Information processing, learning and applications. Acta Neurobiologiae Experimentalis 71(4). https://www.ane.pl/archive?vol=71&no=4&id=7146](https://www.ane.pl/pdf/7146.pdf)
* [Gruening, A, & Bohte, S.M. (2014). Spiking Neural Networks: Principles and Challenges. In Proceedings of Proceeding of the European Symposium on Neural Networks 2014 (ESANN 23).](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-13.pdf)
* [Florian R (2007) Reinforcement Learning Through Modulation of Spike-Timing-Dependent Synaptic Plasticity. Neural Computation 19(6). https://doi.org/10.1162/neco.2007.19.6.1468](https://www.florian.io/papers/2007_Florian_Modulated_STDP.pdf)
* [Frémaux N, Sprekeler H, Gerstner W (2013) Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons. PLOS Computational Biology 9(4): e1003024. https://doi.org/10.1371/journal.pcbi.1003024](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Fremaux13.pdf)

## Package Overview

```none
----------  -----------  ---------  -----
| Neuron |  | Synapse |  | Input |  | ...
----------  -----------  ---------  -----
       \         |         /
         \       |       /
--------   -------------
| Game |   |  Network  |
--------   -------------    ------------
   |____________/___________| Callback | --> Logger --> Reader
   |           /            ------------
-----------------
| Training Loop |
-----------------
        |
----------------------
| Aggregate Analysis |
----------------------
    ^       |
    L_______|
```

Spikey is a spiking neural network framework and training platform.
It provides the components necessary to assemble and configure spiking neural networks
as well as the tools to train them.
There are enough pre-built, parameterized modules to execute many useful experiments out of the box.
Though, it will likely be necessary to write some code in order to pursue a novel goal.

It is important that this platform remains malleable in order to support
users pushing a broad frontier of research, and fast and scaleable enough to
allow for large networks and meta-analysis.
Spikey is written purely in Python with maximal use of
Numpy under the paradigm of array programming.
The malleability of this platform primarily comes from the consistently used flexible
design patterns and well defined input/output schemes.
The speed of this platform is largely achieved with
Numpy, benchmarking and modular code to make the process of optimization straightforward.

Below is a high level overview of the pieces of the
framework and tools provided by the training platform.
See [getting started](#Getting-Started) for usage examples.

### Network

The Network object is the core of the spiking neural network framework.
This module serves as an interface between the environment and the components of the network.
It is configured with a list of parts[a type of synapse, neuron, ...] and a parameter dictionary shared among the given parts.

Find a [usage example here](#Getting-Started).
In order to override the functionality of the network see, [extending 
functionality](#Extending-Functionality). [Network implementation here.](https://github.com/SpikeyCNS/spikey/blob/documentation/spikey/snn/network.py).

### Network Parts

Network parts define how the network will respond to the environment and
learn based on its feedback.
These are the inputs, neurons, synapses, rewarders, readouts and weight matricies.
Each part facilitates the work of the whole group of said parts,
ie, the network only interacts with one neuron part which serves
as an interface for any number of neurons.
This is where the array programming comes into play,
a large amount of work can be done quickly with the smallest
amount of code using numpy. Numpy also scales better than pure python.

Find a [usage example here](#Getting-Started).
In order to create a custom part, see [extending functionality](#Extending-Functionality). [Network part implementations here.](https://github.com/SpikeyCNS/spikey/tree/master/spikey/snn).

### Game

A game is the structure of an environment that defines how agents
can interact with said environment.
In this simulator they serve as an effective, modular way to give input
to and interpret feedback from the network.
A game object is not strictly required for training a network but is highly recommended.

Multiple games have already been made, located in spikey/games/RL for network games and spikey/gamess/MetaRL for meta analysis games.
Find a [usage example here](#Getting-Started).
In order to create new games, see [extending functionality](#Extending-Functionality). [Game implementations here.](https://github.com/SpikeyCNS/spikey/tree/master/spikey/games/RL).

### Callback, Logger and Reader

These are the tools provided for [experiment management](#Experiment-Management). [Logging tool implementations here.](https://github.com/SpikeyCNS/spikey/tree/master/spikey/logging).

### Training Loop

Spikey contains a set of pre-built training loops for network-game interaction
built with the TrainingLoop template.
Though only required for meta-analysis, these will expedite the
development process for many tasks.
On top of that, custom TrainingLoops are extremely easy to share between
experiments and are universally accepted by the tools in this simulator.

See [usage examples in getting started](#Getting-Started). [Training loop implementations here.](https://github.com/SpikeyCNS/spikey/tree/master/spikey/meta).

### Aggregate Analysis

Spikey has tools for running a series of experiments and for hyperparameter searches.
See [meta analysis](#Meta-Analysis) for more detailed information. [Aggregate analysis tool implementations here.](https://github.com/SpikeyCNS/spikey/tree/master/spikey/core).

## Installation

This repository is not yet on PyPi so it must be cloned and installed
locally.

```bash
git clone https://github.com/SpikeyCNS/spikey
cd spikey
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Package

```bash
# Only needs to be run when repo newly cloned or moved, not after edits!
pip install -e .
```

### Run Tests

```bash
pip install -r unit_tests/requirements.txt

pytest  # Unit tests

pytest --nbval-lax  # Unit tests and notebook verification
```

### Build Documentation

```bash
cd docs/
pip install -r requirements.txt
make docs PYTHON3FUNC=<python_function, default=python3>
```

## Getting Started

Many more examples including everything from simple network experiments to hyperparameter tuning can be found in [**examples/**](https://github.com/SpikeyCNS/spikey/tree/master/examples).

## Spikey Features

## Experiment Management

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
Out of the box, this package provides data tracking, analyzing, logging and the corresponding log reading tools.
Each group of tools contains functionality to analyze a single or an aggregate of experiments.

### Tracking Signals

Although it is possible to custom write code to track some variable,
it is best practice to use a callback object.
This will make it easy to track the same signal regardless of training
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
At runtime, either of these options may be extended via ```callback.track("<network/game>_<methodname>", "<results/info>", "<key>", target=["<network/game>", "<part_name>", "<variable_name>"], method="<list/scalar>")```.

[Callback implementations here.](https://github.com/SpikeyCNS/spikey/blob/master/spikey/core/callback.py)

```python
"""
Signal tracking demonstration, creating ndarrays with TD reward data.
"""
import spikey

callback = spikey.core.RLCallback()
callback.track(
    "network_reward",
    "info",
    "td_td",
    ["network", "rewarder", "prev_td"],
    "list"
)
callback.track(
    "network_reward",
    "info",
    "td_reward",
    ["network", "rewarder", "prev_reward"],
    "list",
)
callback.track(
    "network_reward",
    "info",
    "td_value",
    ["network", "rewarder", "prev_value"],
    "list",
)

training_loop = spikey.core.GenericLoop(network_template, game_template, callback, training_params)
network, game, results, info = training_loop()
```

### Logging Data

Functionality exists to store the results of a single experiment via _log_ or
of a series of experiments in separate files via the _MultiLogger_.
Filenames can be given arbitrarily or will be automatically generated in
the form ```YYYY-MM-DD-HH-MM.json``` from _log_ and similarly as ```YYYY-MM-DD-HH-MM-<5_letter_id>.json``` from _MultiLogger_ where the date
is static.
On top of this, the _MultiLogger_ has a summary method which will create a meta
data file with the name ```YYYY-MM-DD-HH-MM~SUMMARY.json```.

All logs are in the json format, which is roughly equivalent to a big python
dictionary.
Each file contains four sections, or subdictionaries: network, game, results and info.
The network and game sections contain the respective module's full configuration.
The results and info subdictionaries come directly from the callback object,
with results for scalar variables easily loaded into tables and info containing ndarrays and generic(serializable) objects.
Before saving to file, each dictionary will be sanitized for json compatibility,
notably ndarrays will be converted to strings - this can be undone via
_uncompressnd_ or the Reader detailed below.

[Logger implementation here.](https://github.com/SpikeyCNS/spikey/blob/master/spikey/logging/log.py)


```python
"""
Data logging demonstration.
"""
import spikey
from spikey.logging import log, MultiLogger

## Single file
training_loop = spikey.core.GenericLoop(
    network_template,
    game_template,
    training_params
)
network, game, results, info = training_loop()
training_loop.log()

## Multiple files
logger = MultiLogger()
for _ in range(10):
    training_loop = spikey.core.GenericLoop(
        network_template,
        game_template,
        training_params
    )
    network, game, results, info = training_loop()

    logger.log(network, game, results, info)

logger.summarize()
```

### Reading logs

A single or multiple log files can be read into memory via the _Reader_ object.
The reader will automatically restore any serialized values to their
original type.

_Reader_ takes two parameters on initialization, the folder to search and a list of
filenames, which if left empty will become all json files in the given folder.
Depending on what section you are looking to pull from, _Reader.df_ may be used
to retrieve a pandas dataframe containing everything from the network, game and
results sections. Otherwise _Reader["\<key\>"]_ / _Reader.\_\_getitem\_\_("\<key\>")_
may be used to retrieve a column from any section.

[Log reader implementation here.](https://github.com/SpikeyCNS/spikey/blob/master/spikey/logging/reader.py)

```python
"""
Reading log data demonstration.
"""
import os
import spikey

reader = spikey.logging.Reader(os.path.join("log", "control"))

print(reader.df["accuracy"])

print(reader["step_states"])
```

### Interpreting logs

A small set of pre-built visualization and analysis functions exists in
_spikey/viz_.
Jupyter notebooks containing even more tools also exist in this repo,
_examples/meta_analysis.ipynb_ and _examples/series_analysis.ipynb_.
If you have made a custom visualization that would be helpful to
others, please submit a feature pull request!

[Viz tool implementations here.](https://github.com/SpikeyCNS/spikey/tree/master/spikey/viz)

## Meta Analysis

```none
--------   -------------
| Game |   |  Network  |
--------   -------------
   |            /
   |           /
-----------------
| Training Loop |
-----------------
        |
----------------------
| Aggregate Analysis |
----------------------
    ^       |
    L_______|
```

Often it is important to run the same experiment multiple times in order to gauge algorithm effectiveness or determine the capability of a parameter.
Other times a hyperparameter search can be used to effectively
solve a problem.
To suit these needs, Spikey provides two meta-analysis tools out of the box as well as the preliminaries for custom distributed meta-analysis modules.
All network parts, training loops and experiment management tools are multiprocessing friendly.

### Multiple Runs and Series of Experiments

The _Series_ tool exists to run multiple experiments in parrallel with
the same or different configurations.
The parameters of this tool are a network and game template, a training loop, a job specification and optionally a callback.

```none
Series Specification
---------------------
configuration = {"name": <details>}

Details can be one of the following,
* (attr, startval=0, stopval, step=1) -> np.arange(*v[1:])
* (attr, [val1, val2...]) -> (i for i in list)
* (attr, generator) -> (i for i in generator)
* [tuple, tuple, ...] -> Each tuple is (attr, list), each list is iterated synchronously.
```

See [series usage example here](https://github.com/SpikeyCNS/spikey/blob/master/examples/run_series.py). [Series implementation here.](https://github.com/SpikeyCNS/spikey/blob/master/spikey/meta/series.py)

### Hyperpater tuning

Spikey has a fully parameterized, extendable genetic algorithm called _Population_.
Similar to _network.config_ and _network._template_parts_,
this has _GENOTYPE\_CONSTRAINTS_ that are best given as
parameters.
_GENOTYPE\_CONSTRAINTS_ defines the range of valid values for each parameter, as
```{genotype: [valid values]}``` or ```{genotype: (range_start, range_stop, range_step)}```.

See [population usage example here](https://github.com/SpikeyCNS/spikey/blob/master/examples/run_meta.py). [Population implementation here.](https://github.com/SpikeyCNS/spikey/blob/master/spikey/meta/population.py)

### Custom Tools

It should be relatively simple to implement custom aggregate analysis tools
with the tools Spikey provides.
The distributed backends used for _Series_ and _Population_ are open for
reuse in _spikey/meta/backends_.
All networks, games and training loops should be friendly to use with these
backends given they are tested with _Population_ and _Series_.

## Extending Functionality

Spikey aims to support users pursuing a broad range of new and niche
spiking neural network related endeavors.
It is of high importance that this platform can be molded to suit many distinct
needs simply and effectively.
Everything in this SNN framework and training suite can be templated, customized
and replaced.
If ever you find it troublesome to implement a certain customization,
please leave the details in the issues tab!

The malleability of this platform primarily comes from the consistently used flexible
design patterns and well defined input/output schemes.

* Parameterization info is stored in class variable that can be extended in
child classes.
* Configuration dictionary loaded to object as ```self._<key> = value``` in
init function.
* Methods are rigidly defined, thus custom parts which respect their
template should be able to comfortably engage with other associated parts.
* Module-module interaction is structured simply, which means a broad range of different part dynamics can interact with each other nicely.

### Instructions

1. Import pertinent module template, ie ```from spikey.<part>.template import <Part>```
2. Create class that inherits template, eg ```class NewSynapse(Synapse):```
3. Override relevant template functions and variables, extend configuration as necessary.
4. (optional) Add custom module to unit tests in file ```unit_tests/test_<part>::Test<Part>.run_all_types```

### Implementation Details in Code

```python
"""
The creation of a neuron.
"""
from spikey.module import Key
from spikey.neuron.template import Neuron


class WillyNilly(Neuron):
    """
    Neuron group firing erratically only.
    """
    ## Instead of completely overwriting the templates NECESSARY_KEYS,
    ## it should be copied and extended.
    NECESSARY_KEYS = Neuron.extend_keys([
        Key("fire_rate", "Rate at which neurons fire.", float, default=.08)
    ])
    def __init__(self, **kwargs):
        # __init__ is optional, but if overriding make sure to call the
        # templates __init__!
        super().__init__(**kwargs)

        # Neuron.__init__ adds all NECESSARY_KEYS to class as member variables
        # prefaced with an underscore(_) ie self._<key> == <value given>.
        print(self._fire_rate)

    def __ge__(self, threshold: float) -> ndarray:
        """
        Overriding Neuron >= threshold which is used to determine
        neuron output at each step.

        Ensure function parameters and returns are compatible with the
        original, unless you really know what you are doing.

        Parameters
        ----------
        threshold: float
            Spiking neuron firing threshold.

        Returns
        -------
        ndarray[n_neurons] Neuron outputs.
        """
        fire_locs = np.random.uniform(0, 1, size=self._n_neurons) <= self._fire_rate

        output = fire_locs * self.polarities * self._magnitude

        return output
```

## Contributing

See our [contributing guide](https://github.com/SpikeyCNS/spikey/blob/master/CONTRIBUTING.md).
