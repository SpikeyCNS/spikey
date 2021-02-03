"""
Core __init__.
"""
try:
    from spikey.core.callback import ExperimentCallback, RLCallback, TDCallback
    from spikey.core.training_loop import TrainingLoop, GenericLoop

except ImportError as e:
    raise ImportError(f"core/__init__.py failed: {e}")
