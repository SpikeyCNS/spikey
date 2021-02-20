# Core Experiment Tools

Core experiment tools are what connect the pieces of the
Spikey together. 
TrainingLoop allows users to train a network on a game individually, repeatedly or within the meta analysis tools.
ExperimentCallback gathers various network, game and general experiment data while training. This is the core of Spikey's experiment management and its format is used by the logger and reader.

There are multiple pre-built implementations of each core tool, though if you are a more advanced user you can template these tools directly and create your own.
