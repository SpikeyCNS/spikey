"""
Tests for meta.backend.
"""
import unittest
from unit_tests import BaseTest
from spikey.meta import backends


def sample_fn(*args):
    return args


class TestBackend(unittest.TestCase, BaseTest):
    """
    Tests for meta.backend.
    """

    TYPES = [backends.SingleProcessBackend, backends.MultiprocessBackend]
    BASE_CONFIG = {}

    @BaseTest.run_all_types
    def test_usage(self):
        backend = self.get_obj()

        for arg_set in [[], [(1, 2, 3)], [(1, 2), (3, 4), (5, 6)]]:
            with self.subTest(arg_set):
                output = backend.distribute(sample_fn, arg_set)
                self.assertListEqual(arg_set, output)


if __name__ == "__main__":
    unittest.main()
