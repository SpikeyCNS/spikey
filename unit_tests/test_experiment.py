"""
Testing the spiking neural network.
"""
import os
from copy import deepcopy
import unittest
from unittest import mock

import numpy as np
import pickle

from spikey.core import RLCallback


class TestExperimentCallback(unittest.TestCase):
    """
    Assorted tests for the snn.
    """

    @staticmethod
    def _get_experiment(*args, **kwargs):
        np.random.seed(0)

        config = {
            "n_episodes": 1,
            "len_episode": 1,
            "reduced": True,
        }
        config.update(kwargs)

        experiment = RLCallback(**config)

        return experiment

    def test_pickle(self):
        """
        Ensure can be pickled.
        """
        experiment = self._get_experiment()
        filename = "experiment_pickle.obj"

        try:
            with open(filename, "wb") as file:
                pickle.dump(experiment, file)

            os.remove(filename)

        except Exception as e:
            os.remove(filename)
            self.fail(e)

    def test_invalid(self):
        """
        Testing experiment.__getattr__.

        Returns
        -------
        lambda *a, *kw: False
        """
        experiment = self._get_experiment()

        output = experiment.test_random_function

        self.assertTrue(callable(output))
        self.assertEqual(output(*list(range(10))), False)

    def test_call(self):
        """
        Testing experiment.__call__.
        """
        ## Standard test
        NETWORK = 123456
        experiment = self._get_experiment()
        experiment.reset()

        experiment.network_init(NETWORK)

        self.assertEqual(experiment.network, NETWORK)

        ## Test w/ custom tracking scalar
        NETWORK = type("Network", (object,), {"test_prop": 122333})

        experiment = self._get_experiment()
        experiment.track(
            "network_init",
            "results",
            "test_name",
            ["network", "test_prop"],
            method="scalar",
        )
        experiment.reset()

        self.assertEqual(experiment.results["test_name"], 0)

        experiment.network_init(NETWORK)

        self.assertEqual(experiment.results["test_name"], 122333)

        ## Test w/ custom tracking list
        NETWORK = type("Network", (object,), {"test_prop": 122333})

        experiment = self._get_experiment()
        experiment.track(
            "network_init",
            "results",
            "test_name",
            ["network", "test_prop"],
            method="list",
        )
        experiment.reset()

        self.assertEqual(experiment.results["test_name"], [])

        experiment.network_reset()
        experiment.network_init(NETWORK)

        self.assertListEqual(experiment.results["test_name"], [[122333]])


if __name__ == "__main__":
    unittest.main()
