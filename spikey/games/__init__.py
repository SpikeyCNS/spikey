"""
Games __init__.
"""
try:
    import spikey.games.RL
    import spikey.games.MetaRL

except ImportError as e:
    raise ImportError(f"games/__init__.py failed: {e}")
