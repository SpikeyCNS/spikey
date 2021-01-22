"""
Print out configuration for each object.
"""
from spikey.network import RLNetwork
from spikey.synapse.template import Synapse
from spikey.neuron.template import Neuron
from spikey.input.template import Input
from spikey.weight.template import Weight
from spikey.reward.template import Reward


def get_necessary_config(silent=False):
    """
    Return all keys that should be in the config dictionary.
    """
    ## NECCESSARY_KEYS = {name: "type description"}
    objs = [RLNetwork, Neuron, Synapse, Input, Weights, Reward]

    if not silent:
        print("Necessary Configuration Keys")

        for obj in objs:
            print(obj.__name__)

            for key, value in obj.NECESSARY_KEYS.items():
                print(f"\t{key}: {value}")

    return {obj.__name__: obj.NECESSARY_KEYS for obj in objs}


if __name__ == "__main__":
    get_necessary_config()
