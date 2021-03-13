"""
Tests for games.
"""
import unittest
from unit_tests import ModuleTest
import gym
from spikey.games import game, RL, MetaRL, gym_wrapper


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

    TYPES = [RL.Logic, RL.CartPole, gym_wrapper(gym.make('CartPole-v0'), base=RL)]
    BASE_CONFIG = {}


'''
class TestRL(unittest.TestCase, ModuleTest):
    """
    Tests for games.MetaRL.
    """

    TYPES = [MetaRL.MetaNQueens, MetaRL.EvolveNetwork, gym_wrapper(gym., base=MetaRL)]
    BASE_CONFIG = {}
'''

if __name__ == "__main__":
    unittest.main()
