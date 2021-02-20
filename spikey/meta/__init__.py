"""
Meta __init__.
"""
try:
    from spikey.meta.series import Series
    from spikey.meta.population import (
        Population,
        GenotypeMapping,
        checkpoint_population,
        read_population,
    )

except ImportError as e:
    raise ImportError(f"meta/__init__.py failed: {e}")
