"""
Tests for meta.Series.
"""
import unittest
from unit_tests import ModuleTest
from spikey.meta.series import Series


class FakeTrainingLoop:
    def __init__(self):
        pass

    def copy(self):
        return self

    def reset(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return "expected_output"


class TestSeries(unittest.TestCase, ModuleTest):
    """
    Tests for meta.Series.
    """

    TYPES = [Series]
    BASE_CONFIG = {
        "training_loop": FakeTrainingLoop(),
        "experiment_params": None,
        "max_process": 1,
        "logging": False,
    }

    @ModuleTest.run_all_types
    def test_usage(self):
        experiment_list = [
            None,
            ("processing_time", [20, 30, 40, 50]),
            ("processing_time", 0, 10),
            [("n_inputs", 5, 25, 5), ("n_neurons", [10, 20, 30])],
        ]

        for series_params in experiment_list:
            with self.subTest(series_params):
                series = self.get_obj(experiment_params=series_params)
                output = series.run(1)
                self.assertTrue(all([value == "expected_output" for value in output]))


if __name__ == "__main__":
    unittest.main()
