"""
Synapse __init__.
"""
try:
    from spikey.snn.synapse.rlstdp import RLSTDP, LTP
    from spikey.snn.synapse.florian import FlorianSTDP
except ImportError as e:
    raise ImportError(f"synapse/__init__.py failed: {e}")
