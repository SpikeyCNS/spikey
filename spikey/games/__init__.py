"""
Games __init__.
"""
try:
    from spikey.games.gym_wrapper import gym_wrapper
    from spikey.games.CartPole import CartPole
    from spikey.games.Logic import Logic

except ImportError as e:
    raise ImportError(f"games/__init__.py failed: {e}")
