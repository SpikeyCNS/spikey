"""
Neuron __init__.
"""
try:
    from spikey.snn.neuron.neuron import Neuron
    from spikey.snn.neuron.rand_potential import RandPotential
except ImportError as e:
    raise ImportError(f"neuron/__init__.py failed: {e}")
