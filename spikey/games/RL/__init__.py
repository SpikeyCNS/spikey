"""
RL __init__.
"""
try:
    from spikey.games.RL.CartPole import CartPole
    from spikey.games.RL.Logic import Logic
except ImportError as e:
    raise ImportError(f"RL/__init__.py failed: {e}")
