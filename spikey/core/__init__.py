"""
Core __init__.
"""
try:
    from spikey.core.callback import ExperimentCallback, RLCallback
    from spikey.core.training_loop import TrainingLoop, GenericLoop, TDLoop

except ImportError as e:
    raise ImportError(f"core/__init__.py failed: {e}")
