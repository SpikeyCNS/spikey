"""
Games __init__.
"""
try:
    import spikey.games.RL
    import spikey.games.MetaRL
    from spikey.games.gym_wrapper import gym_wrapper

except ImportError as e:
    raise ImportError(f"games/__init__.py failed: {e}")
