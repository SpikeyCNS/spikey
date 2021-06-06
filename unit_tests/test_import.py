"""
Test range of accepted spikey imports.
"""
import unittest
from unit_tests import BaseTest


class TestImport(unittest.TestCase, BaseTest):
    """
    Test range of accepted spikey imports.
    """

    def test_base(self):
        import spikey

        spikey.Key

        import spikey.snn
        import spikey.games
        import spikey.core

        import spikey.snn.neuron
        import spikey.games

        from spikey.snn.network import Network
        from spikey.snn.neuron.neuron import Neuron
        from spikey.games.Logic import Logic
        from spikey.meta.population import Population
        from spikey.core.training_loop import TrainingLoop
        from spikey.logging.log import log

    def test_metapath(self):
        """
        Custom paths added by the metapath directory skipper.
        """
        from spikey.snn import RLNetwork
        from spikey.snn.synapse import RLSTDP
        from spikey.reward import MatchExpected
        from spikey.games import CartPole
        from spikey.core import GenericLoop
