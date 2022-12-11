# Spikey Package Directory

The base class for almost all objects in Spikey is the Module located in spikey/module.py.
This provides the general structure and support methods that every object needs.

Each subdirectory in spikey/ contains a README to explain the purpose and usage of the modules it contains.

```none
----------  -----------  ---------  -----
| Neuron |  | Synapse |  | Input |  | ...
----------  -----------  ---------  -----
       \         |         /
         \       |       /
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
