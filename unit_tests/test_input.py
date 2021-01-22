"""
Tests on the input generator.
"""
import unittest

import numpy as np

from spikey.snn.input import *


class TestInput(unittest.TestCase):
    """
    Unit test for input generator.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in [RateMap]:
                with self.subTest(i=obj.__name__):
                    self._get_generator = self._set_obj(obj)

                    func(self)

        return run_all

    def _set_obj(self, obj):
        """
        Create generator that will render only specific object.
        """

        def _get_generator(n_inputs=4, **kwargs):
            np.random.seed(0)

            config = {
                "n_inputs": n_inputs,
                "get_values": lambda state: np.zeros(shape=len(state)),
                "magnitude": 1,
                "firing_steps": -1,
                "rate_mapping": [0, 1],
                "input_pct_inhibitory": 0,
            }

            config.update(kwargs)

            generator = obj(**config)

            return generator

        return _get_generator

    @run_all_types
    def test_len(self):
        """
        Testing input_generator.__len__.

        Returns
        -------
        n_inputs from the generator.
        """
        ## Assert works alone.
        for size in [0, 10, 100]:
            generator = self._get_generator(n_inputs=size)

            self.assertEqual(len(generator), size)

        ## Assert works after update and __call__ use.
        for size in [1, 10, 100]:
            generator = self._get_generator(n_inputs=size)

            generator.update(np.zeros(shape=size))
            generator()

            self.assertEqual(len(generator), size)

    @run_all_types
    def test_call(self):
        """
        Testing input_generator.__call__.

        Settings
        --------
        _magnitude: float
            Spike magnitude.

        Returns
        -------
        Spikes corresponding to generator.values at magnitude _magnitude.
        """
        ## Ensure firing values are respected.
        N_INPUTS = 100
        VALUES = [
            np.ones(shape=N_INPUTS),
            np.zeros(shape=N_INPUTS),
            np.array([int(i % 2) for i in range(N_INPUTS)]),
        ]

        generator = self._get_generator(n_inputs=N_INPUTS, magnitude=1)
        generator.update([1])

        for values in VALUES:
            generator.values = values

            fires = np.float64(generator())
            for i in range(10):
                fires += np.float64(generator())

            self.assertListEqual(
                list(np.where(values, 1, 0)), list(np.where(fires, 1, 0))
            )

        ## Ensure magnitudes are respected.
        N_INPUTS = 100

        for magnitude in [0, 1, 5, 200]:
            generator = self._get_generator(n_inputs=N_INPUTS, magnitude=magnitude)
            generator.update([1])

            generator.values = np.ones(shape=N_INPUTS)

            spikes = generator()

            self.assertListEqual(list(spikes), [magnitude] * N_INPUTS)

        ## Ensure firing_steps respected
        N_INPUTS = 100
        for firing_steps in [0, 5, 8]:
            generator = self._get_generator(
                n_inputs=N_INPUTS, firing_steps=firing_steps
            )
            generator.update([1])

            generator.values = np.ones(shape=N_INPUTS)

            for i in range(15):
                spikes = generator()

                self.assertEqual(generator.network_time, i + 1)

                if i + 1 <= firing_steps:
                    self.assertGreater(np.sum(spikes), 0)
                else:
                    self.assertEqual(np.sum(spikes), 0)

        ## Ensure input_pct_inhibitory respected
        N_INPUTS = 1000
        for input_pct_inhibitory in [0, 0.25, 0.5, 0.75, 1.0]:
            generator = self._get_generator(
                n_inputs=N_INPUTS, input_pct_inhibitory=input_pct_inhibitory
            )
            generator.update([1])

            generator.values = np.ones(N_INPUTS)

            spikes = generator()

            self.assertAlmostEqual(
                np.mean(np.where(spikes < 0, 1, 0)), input_pct_inhibitory, 1
            )

        ## Ensure memory safety
        N_INPUTS = 100

        generator = self._get_generator(n_inputs=N_INPUTS)
        generator.update([1])

        original_values = np.ones(shape=N_INPUTS)
        values = np.copy(original_values)

        generator.values = values

        _ = generator()

        self.assertListEqual(list(original_values), list(values))

    @run_all_types
    def test_update(self):
        """
        Testing input_generator.update

        Parameters
        ----------
        state: np.array(n_inputs)
            Enviornment state.

        Settings
        --------
        _get_values: func
            Function that sets values according to enviornment state.

        Effects
        -------
        Generator values updated by the get_values function applied to state.
        """
        ## Ensure memory safety
        N_INPUTS = 100

        generator = self._get_generator(n_inputs=N_INPUTS)

        original_values = np.ones(shape=N_INPUTS)
        values = np.copy(original_values)

        generator.update(values)

        self.assertListEqual(list(original_values), list(values))


class TestStaticMap(unittest.TestCase):
    """
    Unit test for temporal order input generator.
    """

    @staticmethod
    def _get_generator(**kwargs):
        N_INPUTS = kwargs["n_inputs"] if "n_inputs" in kwargs else 2
        config = {
            "n_inputs": N_INPUTS,
            "get_values": lambda state: np.zeros(shape=len(state)),
            "magnitude": 1,
            "firing_steps": -1,
            "mapping": {(): [0] * N_INPUTS, 0: [0] * N_INPUTS},
            "input_pct_inhibitory": 0,  # unused
        }

        config.update(kwargs)

        generator = StaticMap(**config)

        return generator

    def test_len(self):
        """
        Testing input_generator.__len__.

        Returns
        -------
        n_inputs from the generator.
        """
        ## Assert works alone.
        for size in [0, 10, 100]:
            generator = self._get_generator(n_inputs=size)

            self.assertEqual(len(generator), size)

        ## Assert works after update and __call__ use.
        for size in [1, 10, 100]:
            generator = self._get_generator(n_inputs=size)

            generator.update(())
            generator()

            self.assertEqual(len(generator), size)

    def test_call(self):
        """
        Testing input_generator.__call__.

        Settings
        --------
        _magnitude: float
            Spike magnitude.

        Returns
        -------
        Spikes corresponding to generator.values at magnitude _magnitude.
        """
        ## Ensure firing values are respected.
        N_INPUTS = 3
        MAPPING = {
            (0, 1, 2): np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            (4, 2, 1): np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]]),
            (3, 3, 3): np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]]),
        }

        generator = self._get_generator(n_inputs=N_INPUTS, mapping=MAPPING)

        for key, values in MAPPING.items():
            generator.update([])
            generator.values = key

            for i in range(3):
                fire = generator()

                self.assertListEqual(list(fire), list(values[i]))

        ## Ensure magnitudes are respected.
        N_INPUTS = 100

        for magnitude in [0, 1, 5, 200]:
            generator = self._get_generator(n_inputs=N_INPUTS, magnitude=magnitude)
            generator.update([])

            generator.values = 0

            spikes = generator()

            for value in spikes[np.where(spikes)]:
                self.assertEqual(value, magnitude)

        ## Ensure time working correctly respected
        N_INPUTS = 100
        for firing_steps in [0, 5, 8]:
            generator = self._get_generator(
                n_inputs=N_INPUTS, firing_steps=firing_steps
            )
            generator.update([])

            generator.values = ()

            for i in range(15):
                spikes = generator()

                self.assertEqual(generator.time, i + 1)

    def test_update(self):
        """
        Testing input_generator.update

        Parameters
        ----------
        state: np.array(n_inputs)
            Enviornment state.

        Settings
        --------
        _get_values: func
            Function that sets values according to enviornment state.

        Effects
        -------
        Generator values updated by the get_values function applied to state.
        """
        ## Ensure memory safety
        N_INPUTS = 100

        generator = self._get_generator(n_inputs=N_INPUTS)

        original_values = np.ones(shape=N_INPUTS)
        values = np.copy(original_values)

        generator.update(values)

        self.assertListEqual(list(original_values), list(values))


if __name__ == "__main__":
    unittest.main()
