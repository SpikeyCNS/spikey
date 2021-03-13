"""
Backend __init__.
"""
try:
    from spikey.meta.backends.default import MultiprocessBackend
    from spikey.meta.backends.single import SingleProcessBackend
except ImportError as e:
    raise ImportError(f"meta/backends/__init__.py failed: {e}")
