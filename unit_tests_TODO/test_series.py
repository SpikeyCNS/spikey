"""
Testing the spiking neural network.
"""
from copy import deepcopy
import unittest
from unittest import mock

from spikey.meta import Series
from spikey.snn.network import RLNetwork
from spikey.RL.template import RL
from spikey.core import GenericLoop


class TestSeries(unittest.TestCase):
    """
    Assorted tests for the snn.
    """

    @staticmethod
    def _get_series(experiment_params=None, *args, **kwargs):
        config = {
            "game_config": {"This should be here(game)": True},
            "network_config": {"This should be here(network)": True},
            "This should not be here": True,
        }
        config.update(kwargs)

        network = mock.Mock(spec=RLNetwork)
        network.config = deepcopy(config["network_config"])

        game = mock.Mock(spec=RL)
        game.config = deepcopy(config["game_config"])

        series = Series(GenericLoop, network, game, config, experiment_params)

        return series, game, network

    def test_iter(self):
        """
        Testing series.__iter__.

        Returns
        -------
        Experiment with preconfigured params
        Should not contaminate control SNN/game config!
        """
        ## Ensure returns correct experiments with list given.
        final_i = 0

        experiment_params = ("test_value", [1, 2, 25, "test"])
        EXPECTED = experiment_params[1]

        series, game, network = self._get_series(experiment_params)
        for i, experiment in enumerate(series):
            self.assertEqual(experiment.params["test_value"], EXPECTED[i])

            ## Assert network/game config is not contaminated.
            self.assertDictEqual(game.config, {"This should be here(game)": True})
            self.assertDictEqual(network.config, {"This should be here(network)": True})

            final_i = i

        self.assertEqual(final_i, len(EXPECTED) - 1)

        ## Ensure returns correct experiments with start, stop, step given.
        final_i = 0

        experiment_params = ("test_value", (1, 10, 2))
        EXPECTED = list(range(*experiment_params[1]))

        series, game, network = self._get_series(experiment_params)
        for i, experiment in enumerate(series):
            self.assertEqual(experiment.params["test_value"], EXPECTED[i])

            ## Assert network/game config is not contaminated.
            self.assertDictEqual(game.config, {"This should be here(game)": True})
            self.assertDictEqual(network.config, {"This should be here(network)": True})

            final_i = i

        self.assertEqual(final_i, len(EXPECTED) - 1)

        ## Ensure returns correct experiments with generator given.
        final_i = 0

        experiment_params = ("test_value", (i for i in [1, 2, 25, "test"]))
        EXPECTED = list((i for i in [1, 2, 25, "test"]))

        series, game, network = self._get_series(experiment_params)
        for i, experiment in enumerate(series):
            self.assertEqual(experiment.params["test_value"], EXPECTED[i])

            ## Assert network/game config is not contaminated.
            self.assertDictEqual(game.config, {"This should be here(game)": True})
            self.assertDictEqual(network.config, {"This should be here(network)": True})

            final_i = i

        self.assertEqual(final_i, len(EXPECTED) - 1)

        ## Ensure returns correct experiments with iterable obj given.
        final_i = 0

        ITERABLE = [1, 2, 15, 166]

        class X:
            def __iter__(self):
                for value in ITERABLE:
                    yield value

        experiment_params = ("test_value", X())
        EXPECTED = list(experiment_params[1])

        series, game, network = self._get_series(experiment_params)
        for i, experiment in enumerate(series):
            self.assertEqual(experiment.params["test_value"], EXPECTED[i])

            ## Assert network/game config is not contaminated.
            self.assertDictEqual(game.config, {"This should be here(game)": True})
            self.assertDictEqual(network.config, {"This should be here(network)": True})

            final_i = i

        self.assertEqual(final_i, len(EXPECTED) - 1)

        ## Ensure returns correct experiments with multiple parameters.
        final_i = 0

        experiment_params = [
            ("test_value1", [1, 2, 25, "test"]),
            ("test_value2", [16616, 2, 52, "tes2t"]),
        ]
        EXPECTED = {value[0]: value[1] for value in experiment_params}

        series, game, network = self._get_series(experiment_params)
        for i, experiment in enumerate(series):
            for key, values in EXPECTED.items():
                self.assertEqual(experiment.params[key], values[i])

            ## Assert network/game config is not contaminated.
            self.assertDictEqual(game.config, {"This should be here(game)": True})
            self.assertDictEqual(network.config, {"This should be here(network)": True})

            final_i = i

        self.assertEqual(final_i, len(EXPECTED["test_value1"]) - 1)


if __name__ == "__main__":
    unittest.main()
