"""
unit_tests __init__.
"""
try:
    from unit_tests.base import BaseTest, ModuleTest
except ImportError as e:
    raise ImportError(f"unit_tests/__init__.py failed: {e}")
