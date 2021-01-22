"""
MetaRL __init__.
"""
try:
    from spikey.games.MetaRL.MetaNQueens import MetaNQueens
    from spikey.games.MetaRL.EvolveNetwork import EvolveNetwork
except ImportError as e:
    raise ImportError(f"MetaRL/__init__.py failed: {e}")
