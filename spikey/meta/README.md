# Meta Analysis Tools

Meta analysis tools are for seeing the results of hyper-parameter changes or optimizing hyper-parameters with many experiment runs.
Tool use should be straightforward, though they do require a working TraningLoop with a baseline network and game.

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

## Multiple Runs and Series of Experiments

The _Series_ tool exists to run multiple experiments in parrallel with
the same or different configurations.
The parameters of this tool are a network and game template, a training loop, a job specification and optionally a callback.

```none
Series Specification
---------------------
configuration = {"experiment_name": <details>}

Details can be one of the following,
* (attr, startval=0, stopval, step=1) -> np.arange(*v[1:])
* (attr, [val1, val2...]) -> (i for i in list)
* (attr, generator) -> (i for i in generator)
* [tuple, tuple, ...] -> Each tuple is (attr, list), each list is iterated synchronously.
```

See [series usage example here](https://github.com/SpikeyCNS/spikey/blob/master/examples/run_series.py). [Series implementation here](https://github.com/SpikeyCNS/spikey/blob/master/spikey/meta/series.py).

## Hyperpater tuning

Spikey has a fully parameterized, extendable genetic algorithm called _Population_.
Similar to _network.keys_ and _network.parts_,
this has GENOTYPE_CONSTRAINTS that are best given as
parameters.
GENOTYPE_CONSTRAINTS defines the range of valid values for each parameter, as
```{genotype: [valid values]}``` or ```{genotype: (range_start, range_stop, range_step)}```.

See [population usage example here](https://github.com/SpikeyCNS/spikey/blob/master/examples/run_meta.py). [Population implementation here](https://github.com/SpikeyCNS/spikey/blob/master/spikey/meta/population.py).

## Custom Tools

It should be relatively simple to implement custom aggregate analysis tools
with the tools Spikey provides.
The distributed backends used for _Series_ and _Population_ are open for
reuse in _spikey/meta/backends_.
All networks, games and training loops should be friendly to use with these
backends given they are tested with _Population_ and _Series_.
