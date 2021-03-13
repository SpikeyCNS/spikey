"""
Tests for core.ExperimentCallback.
"""
import unittest
from unit_tests import ModuleTest
from spikey.core import callback


class TestCallback(unittest.TestCase, ModuleTest):
    """
    Tests for core.ExperimentCallback.
    """

    TYPES = [callback.ExperimentCallback, callback.RLCallback]
    BASE_CONFIG = {}

    @ModuleTest.run_all_types
    def test_usage(self):
        callback = self.get_obj()
        bind_name = "test_binding"
        callback.bind(bind_name)

        callback.track(bind_name, "results", "test_scalar", ['arg_0'], "scalar")
        callback.track(bind_name, "info", "test_list", ['arg_0'], "list")
        callback.reset()

        getattr(callback, bind_name)(0)
        self.assertEqual(callback.results["test_scalar"], 0)
        self.assertListEqual(callback.info["test_list"], [0])

        callback.reset()
        self.assertEqual(callback.results["test_scalar"], 0)
        self.assertListEqual(callback.info["test_list"], [])

        for i in range(5):
            getattr(callback, bind_name)(i)
        self.assertEqual(callback.results["test_scalar"], 4)
        self.assertListEqual(callback.info["test_list"], list(range(5)))


if __name__ == "__main__":
    unittest.main()
