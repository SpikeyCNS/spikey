# Games

A game is the structure of an environment that defines how agents
can interact with said environment.
In this simulator they serve as an effective, modular way to give input
to and interpret feedback from the network.

All games are structured similarly to environments in OpenAI's gym. RL games are nearly exactly the same and should be used to train spiking networks, ideally in a TrainingLoop. MetaRL games are designed for the genetic algorithm(Population) which notably has a game to optimize hyperparameters of a Network-RL experiment.
