"""
Tests for base module.
"""
import unittest
from unit_tests import ModuleTest
from spikey import Key, Module


class TestModule(unittest.TestCase, ModuleTest):
    """
    Tests for spikey.Module.
    """

    TYPES = [Module]
    BASE_CONFIG = {}

    @ModuleTest.run_all_types
    def test_extend(self):
        ExtendModule = type("ExtendModule", (Module,), {})

        ExtendModule.NECESSARY_KEYS = {"a": 1, "b": 2}
        keys = ExtendModule.extend_keys({"c": 3})
        for key in ["a", "b", "c"]:
            with self.subTest(key):
                self.assertIn(key, keys)

        ExtendModule.NECESSARY_KEYS = [Key("a", "1"), Key("b", "2")]
        keys = ExtendModule.extend_keys([Key("c", "3")])
        for key in ["a", "b", "c"]:
            with self.subTest(key):
                self.assertIn(key, keys)

        ExtendModule.NECESSARY_KEYS = {"a": 1, "b": 2}
        keys = ExtendModule.extend_keys([Key("c", "3")])
        for key in ["a", "b", "c"]:
            with self.subTest(key):
                self.assertIn(key, keys)

        ExtendModule.NECESSARY_KEYS = [Key("a", "1"), Key("b", "2")]
        keys = ExtendModule.extend_keys({"c": 3})
        for key in ["a", "b", "c"]:
            with self.subTest(key):
                self.assertIn(key, keys)


if __name__ == "__main__":
    unittest.main()
