"""
Test spikey logging functionality.
"""
import unittest
from unit_tests import BaseTest
import os
import json
import numpy as np
from spikey.logging import log, MultiLogger, Reader, sanitize, serialize


class TestLog(unittest.TestCase, BaseTest):
    """
    Test spikey logging functionality.
    """

    FILENAME = "test_log.json"

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(TestLog.FILENAME)
        except FileNotFoundError:
            pass

    def test_sanitize(self):
        for dictionary in [
            None,
            12.3,
            (2, 4),
            {},
            {"one": 123, 24: 25},
            {"obj": type("Obj", (object,), {})},
            {"np.int": np.int_(12)},
        ]:
            output = sanitize(dictionary)
            json_output = json.loads(json.dumps(output))

    def test_serialize(self):
        for shape in [0, 1, 100, (10, 10), (10, 10, 10)]:
            for original in [
                np.ones(shape, dtype=np.int),
                np.random.uniform(0, 1, shape),
                np.zeros(shape, dtype=np.bool),
            ]:
                with self.subTest(f"{original.dtype}, shape={shape}"):
                    compressed = serialize.compressnd(original)
                    output = serialize.uncompressnd(compressed)
                    np.testing.assert_array_equal(original, output)

    def test_multilog(self):
        multilogger = MultiLogger(".")
        prev = []
        for i, filename in enumerate(multilogger.filename_gen()):
            self.assertNotIn(filename, prev)
            prev.append(filename)
            if i >= 100:
                break

    def test_reader(self):
        for results in [{}, {"column1": 23}]:
            with self.subTest(f"{results}"):
                log(results=results, filename=self.FILENAME)
                Reader(".", [self.FILENAME])
                Reader(self.FILENAME)

    def test_usage(self):
        network = type(
            "SpoofNetwork", (object,), {"parts": {}, "params": {"p1": 1, "p2": 2.3}}
        )
        game = type("SpoofRL", (object,), {"params": {"name": 0, "p1": 1, "p2": 2.3}})
        results = {"r1": 1, "r2": 2.3}
        info = {"i1": [1, 2, 3], "i2": (4, 5), "i3": np.random.uniform(-1, 1, size=5)}
        for kwparams in [
            {"network": network, "game": game, "results": results, "info": info},
            {"network": network, "game": game, "results": results},
            {"network": network, "game": game},
            {"results": results, "info": info},
        ]:
            with self.subTest(f"{kwparams}"):
                log(filename=self.FILENAME, **kwparams)
                reader = Reader(self.FILENAME)
                if "network" in kwparams:
                    np.testing.assert_array_equal(
                        list(network.params.values()), reader.network.values[0]
                    )
                if "game" in kwparams:
                    np.testing.assert_array_equal(
                        list(game.params.values()), reader.game.values[0]
                    )
                if "results" in kwparams:
                    np.testing.assert_array_equal(
                        list(results.values()), reader.results.values[0]
                    )
                if "info" in kwparams:
                    for key, value in info.items():
                        np.testing.assert_array_equal(value, reader[key][0])


if __name__ == "__main__":
    unittest.main()
