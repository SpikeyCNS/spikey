"""
Synapse __init__.
"""
try:
    from spikey.snn.synapse.rlstdp import RLSTDP, LTP
except ImportError as e:
    raise ImportError(f"synapse/__init__.py failed: {e}")
