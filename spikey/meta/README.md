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
The distributed backend used for _Population_ is open for
reuse in _spikey/meta/backends_.
All networks, games and training loops should be friendly to use with these
backends given they are tested with _Population_.
