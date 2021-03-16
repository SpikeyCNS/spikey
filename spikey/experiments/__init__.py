"""
Experiment __init__.py
"""
try:
    import spikey.experiments.benchmark
    import spikey.experiments.florian_rate
    import spikey.experiments.florian_temporal

except ImportError as e:
    raise ImportError(f"experiments/__init__.py failed: {e}")
