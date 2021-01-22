"""
Reward __init__.
"""
try:
    from spikey.snn.reward.match_expected import MatchExpected
    from spikey.snn.reward.tderror import TDError
except ImportError as e:
    raise ImportError(f"reward/__init__.py failed: {e}")
