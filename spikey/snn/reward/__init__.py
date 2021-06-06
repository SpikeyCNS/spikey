"""
Reward __init__.
"""
try:
    from spikey.snn.reward.match_expected import MatchExpected
except ImportError as e:
    raise ImportError(f"reward/__init__.py failed: {e}")
