"""
Test modifier functions.
"""
import unittest

import numpy as np

from spikey.snn.modifier import *


##
class TestNeuron:
    __test__ = False

    def __init__(self):
        self.y = 0


class TestNet:
    __test__ = False

    def __init__(self):
        self.x = 0

        self.neurons = TestNeuron()


##
from spikey.snn.modifier.template import Modifier


class FakeModifier(Modifier):
    pass


##
class TestModifier(unittest.TestCase):
    """
    Modifier function tests.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in [LinearDecay, DropOff]:
                with self.subTest(i=obj.__name__):
                    self._get_modifier = self._set_obj(obj)

                    func(self)

        return run_all

    def _set_obj(self, obj):
        """
        Create generator that will render only specific object.
        """

        def _get_modifier(*args):
            np.random.seed(0)

            modifier = obj(*args)

            return modifier

        return _get_modifier

    @run_all_types
    def test_eq(self):
        """
        Testing modifier.__eq__.

        Parameters
        ----------
        other: modifier
            Modifier object to compare with.

        Returns
        -------
        bool Whether is equal to or not.
        """
        ## Ensure only works if same type
        modifier1 = self._get_modifier([], 1, 2, 3)
        modifier2 = FakeModifier

        self.assertFalse(modifier1 == modifier2)

        #
        modifier1 = self._get_modifier([], 1, 2, 3)
        modifier2 = self._get_modifier([], 1, 2, 3)

        self.assertTrue(modifier1 == modifier2)

        ## Ensure only works if params are same
        modifier1 = self._get_modifier([], 1, 2, 3)
        for args in [
            (
                6,
                2,
                3,
            ),
            (3, 2, 1),
            (187, 1676, 1616),
        ]:
            modifier2 = self._get_modifier([], *args)

            self.assertFalse(modifier1 == modifier2)

        #
        modifier1 = self._get_modifier([], 1, 2, 3)
        modifier2 = self._get_modifier([], 1, 2, 3)

        self.assertTrue(modifier1 == modifier2)

    @run_all_types
    def test_set_param(self):
        """
        Testing modifier.set_param.

        Parameters
        ----------
        network: SNN
            Network to update values for.
        value: any
            Value to update parameter with.

        Settings
        --------
        param: list[str]
            List of values

        Effects
        -------
        Value should be updated.
        """
        ## Ensure correct value is set
        PARAM = ["network", "x"]

        for value in [0.123, 100, 61.16]:
            modifier = self._get_modifier(PARAM, 1, 2, 3)
            network = TestNet()

            modifier.set_param(network, value)

            self.assertEqual(network.x, value)
            self.assertEqual(network.neurons.y, 0)

        PARAM = ["network", "neurons", "y"]

        for value in [0.123, 100, 61.16]:
            modifier = self._get_modifier(PARAM, 1, 2, 3)
            network = TestNet()

            modifier.set_param(network, value)

            self.assertEqual(network.x, 0)
            self.assertEqual(network.neurons.y, value)


if __name__ == "__main__":
    unittest.main()
