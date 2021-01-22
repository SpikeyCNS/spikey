"""
Weight matrix __init__.
"""
try:
    from spikey.snn.weight.manual import Manual
    from spikey.snn.weight.random import Random
except ImportError as e:
    raise ImportError(f"w/__init__.py failed: {e}")
