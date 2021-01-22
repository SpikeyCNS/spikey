"""
Input __init__.
"""
try:
    from spikey.snn.input.ratemap import RateMap
    from spikey.snn.input.staticmap import StaticMap
    from spikey.snn.input.rbf import RBF
    from spikey.snn.input.track_rbf import TrackRBF
except ImportError as e:
    raise ImportError(f"input/__init__.py failed: {e}")
