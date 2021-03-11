"""
Test range of accepted spikey imports.
"""
import unittest
import unit_tests


class TestImport(unittest.TestCase, unit_tests.BaseTest):
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
        import spikey.games.RL

        from spikey.snn.network import Network
        from spikey.snn.neuron.neuron import Neuron
        from spikey.games.RL.Logic import Logic
        from spikey.meta.population import Population
        from spikey.core.training_loop import TrainingLoop
        from spikey.logging.log import log

    def test_metapath(self):
        """
        Custom paths added by the metapath directory skipper.
        """
        from spikey.snn import RLNetwork
        from spikey.snn.synapse import RLSTDPET
        from spikey.reward import MatchExpected
        from spikey.RL import CartPole
        from spikey.core import GenericLoop
