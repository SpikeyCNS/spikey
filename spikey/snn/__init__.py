"""
SNN __init__.py
"""
try:
    import spikey.snn.input
    import spikey.snn.modifier
    import spikey.snn.neuron
    import spikey.snn.readout
    import spikey.snn.reward
    import spikey.snn.synapse
    import spikey.snn.weight

    from spikey.snn.network import Network, RLNetwork, ActiveRLNetwork

except ImportError as e:
    raise ImportError(f"snn/__init__.py failed: {e}")
