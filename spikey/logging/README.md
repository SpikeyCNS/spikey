# Logging and File Reading Tools

spikey.logging contains all data log reading and writing tools.

## Writing Logs

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
Results and info are separate dictionaries, with results for scalar variables easily
loaded into tables and info containing ndarrays and generic(serializable) objects.
Before saving to file, each dictionary will be sanitized for json compatibility,
notably ndarrays will be converted to strings - this can be undone via
_uncompressnd_ or the Reader detailed below.

```python
"""
Data logging demonstration.
"""
import spikey
from spikey.logging import log, MultiLogger

# Single file
training_loop = spikey.core.GenericLoop(
    network_template,
    game_template,
    training_params
)
network, game, results, info = training_loop()
training_loop.log()

# Multiple files
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

## Reading logs

A single or multiple log files can be read into memory via the _Reader_ object.
The reader will automatically restore any serialized values to their
original type.

_Reader_ takes two parameters on initialization, the folder to search and a list of
filenames, which if left empty will become all json files in the given folder.
Depending on what section you are looking to pull from, _Reader.df_ may be used
to retrieve a pandas dataframe containing everything from the network, game and
results sections. Otherwise _Reader["key"]_ / _Reader_.\_\_getitem\_\_("key")
may be used to retrieve a column from any section.

[Log reader implementation here](https://github.com/SpikeyCNS/spikey/blob/master/spikey/logging/reader.py).

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
