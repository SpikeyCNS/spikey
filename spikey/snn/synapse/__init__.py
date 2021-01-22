"""
Synapse __init__.
"""
try:
    from spikey.snn.synapse.rlstdpet import RLSTDPET
    from spikey.snn.synapse.ltp import LTP
except ImportError as e:
    raise ImportError(f"synapse/__init__.py failed: {e}")
