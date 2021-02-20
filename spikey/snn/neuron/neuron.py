"""
A group of spiking neurons.

Each spiking neuron has an internal membrane potential that
increases with each incoming spike. The potential persists but slowly
decreases over time. Each neuron fires when its potential surpasses
some firing threshold and does not fire again for the duration
of its refractory period.
"""
from spikey.snn.neuron.template import Neuron
