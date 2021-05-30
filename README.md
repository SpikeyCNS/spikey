# Spikey

Spikey is a malleable, [ndarray based](https://numpy.org/doc/stable/reference/arrays.ndarray.html) spiking neural network framework and training platform. It contains many pre-made components, experiments and meta-analysis tools(genetic algorithm). It's regularly tested with Python 3.7-3.9 in Linux and Windows. Expect to post bugs or suggestions in the issues tab :)

## Table of Contents

* [Spiking Neural Networks](#spiking-neural-networks)
* [Package Overview](#package-overview)
* [Installation](#installation)
* [Getting Started](#getting-started)
* [Contributing](#contributing)

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
See [usage example here](#getting-started).

### Network

The Network object is the core of the spiking neural network framework.
This module serves as an interface between the environment and the components of the network.
It is configured with a list of parts[a type of synapse, neuron, ...] and a parameter dictionary shared among the given parts.

Network parts define how the network will respond to the environment and
learn based on its feedback.
These are the inputs, neurons, synapses, rewarders, readouts and weight matricies.
Each part facilitates the work of the whole group of said parts,
ie, the network only interacts with one neuron part which serves
as an interface for any number of neurons.
This is where the array programming comes into play,
a large amount of work can be done quickly with the smallest
amount of code using numpy. Numpy also scales better than pure python.

Find a [usage example here](#getting-started).
In order to override the functionality of the network see, [extending functionality](https://github.com/SpikeyCNS/spikey/blob/master/spikey/snn/README.md#extending-functionality). [Network implementation here](https://github.com/SpikeyCNS/spikey/blob/master/spikey/snn/network.py).

### Game

A game is the structure of an environment that defines how agents
can interact with said environment.
In this simulator they serve as an effective, modular way to give input
to and interpret feedback from the network.
A game object is not strictly required for training a network but is highly recommended.

Multiple games have already been made, located in spikey/games/RL for network games and spikey/gamess/MetaRL for meta analysis games.
Find a [usage example here](#getting-started).
In order to create new games, see [extending functionality](https://github.com/SpikeyCNS/spikey/blob/master/spikey/snn/README.md#extending-functionality). [Game implementations here](https://github.com/SpikeyCNS/spikey/tree/master/spikey/games/RL).

### Callback, Logger and Reader

These are the tools provided for experiment analysis. [Logging tool implementations here](https://github.com/SpikeyCNS/spikey/tree/master/spikey/logging).
[Callback implementations here](https://github.com/SpikeyCNS/spikey/tree/master/spikey/core).

### Training Loop

Spikey contains a set of pre-built training loops for network-game interaction
built with the TrainingLoop template.
Though only required for meta-analysis, these will expedite the
development process for many tasks.
On top of that, custom TrainingLoops are extremely easy to share between
experiments and are universally accepted by the tools in this simulator.

See [usage example here](#getting-started). [Training loop implementations here](https://github.com/SpikeyCNS/spikey/tree/master/spikey/core).

### Aggregate Analysis

Spikey has tools for running a series of experiments and for hyperparameter searches.
[Aggregate analysis tool implementations here](https://github.com/SpikeyCNS/spikey/tree/master/spikey/meta).

## Installation

This repository is not yet on PyPi so it must be cloned and installed
locally. It only needs to be installed when the repo newly cloned or moved, not after code edits!

```bash
git clone https://github.com/SpikeyCNS/spikey
cd spikey
pip install -e .
```

### Run Tests

```bash
pip install -r unit_tests/requirements.txt

# If use python command
bash unit_tests/run.sh
# Else if use python3 command
bash unit_tests/run3.sh
```

## Getting Started

Many more examples including everything from simple network experiments to hyperparameter tuning can be found in [examples/](https://github.com/SpikeyCNS/spikey/tree/master/examples).

## Contributing

See our [contributing guide](https://github.com/SpikeyCNS/spikey/blob/master/CONTRIBUTING.md).
