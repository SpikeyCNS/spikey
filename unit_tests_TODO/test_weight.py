"""
Tests on the weight matrix generator.
"""
import unittest

import numpy as np

import spikey.snn.weight as weight


class TestRandom(unittest.TestCase):
    """
    Unit test for input generator.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for wfunc in [
                weight.Random,
            ]:
                with self.subTest(i=wfunc.__name__):
                    self._get_wfunc = self._set_func(wfunc)

                    func(self)

        return run_all

    def _set_func(self, func):
        """
        Create generator that will render only specific function.
        """

        def _get_wfunc(**kwargs):
            np.random.seed(0)

            config = {
                "n_neurons": 10,
                "n_inputs": 4,
                "max_weight": 1,
                "force_unidirectional": False,
                "matrix_mask": None,
                "inh_weight_mask": None,
                "weight_generator": lambda shape: np.random.uniform(size=shape),
            }
            config.update(kwargs)

            return func(**config)

        return _get_wfunc

    @run_all_types
    def test_init(self):
        ## Ensure all masked values have 0 in the .data
        w = self._get_wfunc()

        # if np.where(masked) == .data == 0
        if isinstance(w._matrix, np.ma.MaskedArray):
            self.assertListEqual(
                list(np.where(np.ravel(w._matrix.mask))[0]),
                list(np.where(np.ravel(w._matrix.data) == 0)[0]),
            )

        self.assertIsInstance(w._matrix, np.ma.MaskedArray)

        ## Ensure respects weight_generator
        w = self._get_wfunc(
            n_inputs=0,
            n_neurons=100,
            max_weight=1,
            weight_generator=np.ones,
        )
        self.assertTrue(np.all(w._matrix == 1))

        w = self._get_wfunc(
            n_inputs=0,
            n_neurons=100,
            weight_generator=np.zeros,
        )
        self.assertTrue(np.all(w.matrix == 0))

        ## Ensure responds to force_unidirectional
        w = self._get_wfunc(
            n_inputs=0,
            n_neurons=100,
            force_unidirectional=True,
            weight_generator=np.ones,
        )

        for y in range(w._matrix.shape[0]):
            for x in range(y, w._matrix.shape[1]):
                self.assertFalse(w._matrix[y, x] and w._matrix[x, y])
                self.assertFalse(not w._matrix.mask[y, x] and not w._matrix.mask[x, y])

    @run_all_types
    def test_io(self):
        """
        Testing weight matrix generator.

        Returns
        -------
        weight matrix, n_inputs+n_neurons x n_neurons
        """
        ## Assert output matrix is expected size and
        for n_inputs in range(0, 10, 2):
            for n_neurons in range(4, 10, 2):
                w = self._get_wfunc(n_inputs=n_inputs, n_neurons=n_neurons)

                self.assertEqual(w.matrix.shape[0], n_inputs + n_neurons)
                self.assertEqual(w.matrix.shape[1], n_neurons)

                diagonal = np.arange(n_neurons)
                self.assertFalse(np.sum(w[n_inputs + diagonal, diagonal]))

        ## Assert fill values are correct
        try:
            w = self._get_wfunc(n_inputs=10, n_neurons=10)
            self.assertEqual(w._matrix.fill_value, 0)
        except AttributeError:
            pass

    @run_all_types
    def test_arithmetic(self):
        """
        Testing inplace/normal arithmetic.

        Returns
        -------
        Nothing or updated weight matrix.
        """

        def to_list(ndarray):
            if ndarray.dtype == "bool":
                return list(np.ravel(ndarray))

            return [list(row) for row in np.round(ndarray, decimals=4)]

        has_mask = True
        try:
            self._get_wfunc().mask
        except AttributeError:
            has_mask = False

        ## Assert all standard work
        for method in ["__add__", "__sub__", "__truediv__", "__mul__"]:
            max_weight = 100
            w = self._get_wfunc(max_weight=max_weight)
            original = w.matrix.copy()
            if has_mask:
                static_mask = np.copy(w._matrix.mask)

            for value in [1, 2, 5]:
                self.assertListEqual(
                    to_list(getattr(w, method)(value)),
                    to_list(getattr(original, method)(value)),
                )
                if has_mask:
                    self.assertListEqual(to_list(w._matrix.mask), to_list(static_mask))

        ## Assert all inplace work
        for method in ["__iadd__", "__isub__", "__itruediv__", "__imul__"]:
            max_weight = 100
            w = self._get_wfunc(max_weight=max_weight)
            original = w._matrix.copy()
            if has_mask:
                static_mask = np.copy(w._matrix.mask)

            for value in [1, 2, 5]:
                getattr(w, method)(value)
                getattr(original, method)(value)
                original = np.clip(original, 0, max_weight)

                self.assertListEqual(to_list(w._matrix), to_list(original))
                if has_mask:
                    self.assertListEqual(to_list(w._matrix.mask), to_list(static_mask))

            self.assertIsInstance(w._matrix, np.ma.MaskedArray)


class TestManual(unittest.TestCase):
    def get_matrix(self, **kwargs):
        config = {
            "n_neurons": 100,
            "n_inputs": 50,
            "max_weight": 1,
        }
        config.update(kwargs)

        return weight.Manual(**config)

    def test_init(self):
        ## Ensure memory safe when matrix is array
        N_NEURONS = 100
        N_INPUTS = 0

        original_w = np.ones((N_NEURONS, N_NEURONS))
        w = np.copy(original_w)
        original_inh = np.zeros((N_NEURONS, N_NEURONS))
        inh = np.copy(original_inh)

        W = self.get_matrix(
            matrix=w,
            inh_weight_mask=inh,
            max_weight=100,
            n_neurons=N_NEURONS,
            n_inputs=N_INPUTS,
        )
        W._matrix += 33

        self.assertListEqual(list(np.ravel(W.matrix)), list(np.ravel(w + 33)))
        self.assertListEqual(list(np.ravel(original_w)), list(np.ravel(w)))
        self.assertListEqual(list(np.ravel(W._inh_weight_mask)), list(np.ravel(inh)))

    def test_mul(self):
        ## Ensure polarities respected
        N_NEURONS = 20
        N_INPUTS = 0

        for _ in range(3):
            inh = np.int_(np.random.uniform(0, 1, size=(N_NEURONS, N_NEURONS)) <= 0.5)
            w = np.ones((N_NEURONS, N_NEURONS))

            W = self.get_matrix(
                matrix=np.copy(w),
                inh_weight_mask=np.copy(inh),
                n_neurons=N_NEURONS,
                n_inputs=N_INPUTS,
            )

            output = W * 1

            self.assertListEqual(
                list(np.ravel(output)), list(np.ravel(np.where(inh == 1, -1, 1)))
            )


if __name__ == "__main__":
    unittest.main()
