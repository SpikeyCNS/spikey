"""
Tests for modifier.
"""
import unittest
from unit_tests import ModuleTest
import numpy as np
from spikey.snn import modifier


class TestModifier(unittest.TestCase, ModuleTest):
    """
    Tests for modifier.
    """

    TYPES = [modifier.Modifier]
    BASE_CONFIG = {"param": "processing_time"}


if __name__ == "__main__":
    unittest.main()
