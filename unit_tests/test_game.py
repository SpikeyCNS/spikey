"""
Tests for games.
"""
import unittest
from unit_tests import ModuleTest
from copy import deepcopy
import gym
from spikey import Key
from spikey.games import game, RL, MetaRL, gym_wrapper


class FakeTrainingLoop:
    def __init__(self):
        pass

    def copy(self):
        return self

    def reset(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return "expected_output"


def fitness_getter(*a):
    return 0


class TestGame(unittest.TestCase, ModuleTest):
    """
    Tests for games.Game.
    """

    TYPES = [game.Game]
    BASE_CONFIG = {}


class TestRL(unittest.TestCase, ModuleTest):
    """
    Tests for games.RL.
    """

    TYPES = [RL.Logic, RL.CartPole]  # , gym_wrapper(gym.make('CartPole-v0'), base=RL)]
    BASE_CONFIG = {}

    @ModuleTest.run_all_types
    def test_init(self):
        game_type = type(self.get_obj())

        class game_template(game_type):
            NECESSARY_KEYS = game_type.extend_keys(
                [
                    Key("a", "a"),
                    Key("b", "b"),
                ]
            )
            config = deepcopy(self.BASE_CONFIG)
            config.update({"a": 10, "b": 20})

        a = 11
        game = game_template(a=a)
        self.assertEqual(game.params["a"], a)
        self.assertEqual(game.params["b"], game_template.config["b"])

    @ModuleTest.run_all_types
    def test_usage(self):
        game = self.get_obj()

        state = game.reset()
        for _ in range(100):
            state, reward, done, info = game.step(0)
            if done:
                break

        game.close()


class TestMetaRL(unittest.TestCase, ModuleTest):
    """
    Tests for games.MetaRL.
    """

    TYPES = [
        MetaRL.MetaNQueens,
        MetaRL.EvolveNetwork,
    ]  # , gym_wrapper(gym., base=MetaRL)]
    BASE_CONFIG = {
        "training_loop": FakeTrainingLoop(),
        "genotype_constraints": {"a": [1, 2, 3]},
        "win_fitness": 1,
        "fitness_getter": fitness_getter,
    }

    @ModuleTest.run_all_types
    def test_init(self):
        game_type = type(self.get_obj())

        class game_template(game_type):
            NECESSARY_KEYS = game_type.extend_keys(
                [
                    Key("a", "a"),
                    Key("b", "b"),
                ]
            )
            config = deepcopy(self.BASE_CONFIG)
            config.update({"a": 10, "b": 20})

        a = 11
        game = game_template(a=a)
        self.assertEqual(game.params["a"], a)
        self.assertEqual(game.params["b"], game_template.config["b"])


if __name__ == "__main__":
    unittest.main()
