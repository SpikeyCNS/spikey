"""
Modifier __init__.
"""
try:
    from spikey.snn.modifier.drop_off import DropOff
    from spikey.snn.modifier.linear_decay import LinearDecay
except ImportError as e:
    raise ImportError(f"modifier/__init__.py failed: {e}")
