"""
Meta __init__.
"""
try:
    from spikey.meta.population import (
        Population,
        GenotypeMapping,
        checkpoint_population,
        read_population,
    )
    from spikey.meta.metagames import MetaNQueens, EvolveNetwork

except ImportError as e:
    raise ImportError(f"meta/__init__.py failed: {e}")
