# Spiking Neural Network Framework

SNN/ contains the core spiking neural network framework. This includes each piece of a network: neuron, synapse, ... There are multiple pre-built implementations for each part as well the power to template and customize them all. Though each can be used and interacted with individually, the Network(, RLNetwork, ActiveRLNetwork) object serves as a manager and neat interface for the whole spiking neural network with all of its parts.

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
4. (optional) Add custom module to unit tests in file ```unit_tests/test_<part>::Test<Part>.TYPES```

### Implementation Details in Code

```python
"""
The creation of a neuron.
"""
import numpy as np
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

    def __ge__(self, threshold: float) -> np.ndarray:
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
