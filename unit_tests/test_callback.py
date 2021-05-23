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

        callback.monitor(bind_name, "results", "test_scalar", ["arg_0"], "scalar")
        callback.monitor(bind_name, "info", "test_list", ["arg_0"], "list")
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

    def test_reset(self):
        self.get_obj = self._set_obj(callback.RLCallback)

        c1 = self.get_obj()
        c1.monitor("network_tick", "results", "reset_test", "arg_0", "scalar")
        c2 = c1.copy()
        c1.reset()
        c2.reset()
        c2.network_reset()
        c2.network_tick(200, 200)
        self.assertEqual(len(c1.info["episode_lens"]), 0)
        self.assertEqual(len(c2.info["episode_lens"]), 1)
        self.assertEqual(c1.results["reset_test"], 0)
        self.assertEqual(c2.results["reset_test"], 200)

        # Calling reset then copying, copied version has habit of referencing original on network_tick
        c1 = self.get_obj()
        c1.monitor("network_tick", "results", "reset_test", "arg_0", "scalar")
        c1.reset()
        c2 = c1.copy()
        c2.reset()
        c2.network_reset()
        c2.network_tick(200, 200)
        self.assertEqual(len(c1.info["episode_lens"]), 0)
        self.assertEqual(len(c2.info["episode_lens"]), 1)
        self.assertEqual(c1.results["reset_test"], 0)
        self.assertEqual(c2.results["reset_test"], 200)


if __name__ == "__main__":
    unittest.main()
