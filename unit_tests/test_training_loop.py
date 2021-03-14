"""
Tests for core.TrainingLoop.
"""
import unittest
from unit_tests import ModuleTest
from spikey.core import training_loop


class FakeBase:
    def __init__(self, **kwargs):
        pass

    def __getattr__(self, key):
        return lambda *a, **kw: None


class FakeNetwork(FakeBase):
    pass


class FakeGame(FakeBase):
    def step(self, action):
        return None, 0, False, {}


class FakeCallback:
    def __init__(self, **kwargs):
        pass

    def reset(self, *a, **kw):
        pass

    def training_end(self):
        pass

    def __iter__(self):
        yield "expected_output"


class TestTrainingLoop(unittest.TestCase, ModuleTest):
    """
    Tests for core.TrainingLoop.
    """

    TYPES = [training_loop.GenericLoop]
    BASE_CONFIG = {
        "network_template": FakeNetwork,
        "game_template": FakeGame,
        "callback": FakeCallback,
        "n_episodes": 10,
        "len_episode": 100,
    }

    def run_usage(self, training_loop):
        training_loop.reset()

        for _ in range(5):
            output = training_loop()
            self.assertEqual(output, ["expected_output"])

    @ModuleTest.run_all_types
    def test_init(self):
        training_loop = self.get_obj(callback=FakeCallback)
        self.run_usage(training_loop)

        training_loop = self.get_obj(callback=FakeCallback())
        self.run_usage(training_loop)

    @ModuleTest.run_all_types
    def test_usage(self):
        training_loop = self.get_obj()
        self.run_usage(training_loop)


if __name__ == "__main__":
    unittest.main()
