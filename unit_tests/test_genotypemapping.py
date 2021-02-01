"""
Evaluating the meta rl functionality.
"""
from copy import deepcopy
import unittest

from spikey.meta import GenotypeMapping
from spikey.modifier import DropOff, LinearDecay


class TestGenotypeMapping(unittest.TestCase):
    """
    Testing meta rl GenotypeMapping class.
    """

    def _get_mapping(self, **kwargs):
        config = {
            "n_storing": 100,
        }
        config.update(kwargs)

        mapping = GenotypeMapping(**config)

        return mapping

    def test_update(self):
        """
        Testing GenotypeMapping.update.

        Parameters
        ----------
        genotype: genotype
            Genotype to map.
        fitness: int
            Value to map genotype to.

        Settings
        --------
        n_storing: int
            Number of items to store.
        """
        ## Ensure correct number of values stored
        for n_storing in [0, 5, 12]:
            mapping = self._get_mapping(n_storing=n_storing)

            for i in range(20):
                genotype = {"idx": i}

                mapping.update(genotype, i)

                self.assertLessEqual(len(mapping.genotypes), n_storing)
                self.assertLessEqual(len(mapping.fitnesses), n_storing)

            self.assertEqual(len(mapping.genotypes), n_storing)
            self.assertEqual(len(mapping.fitnesses), n_storing)

        ## Ensure correct ordering maintained
        for n_storing in [0, 5, 12]:
            mapping = self._get_mapping(n_storing=n_storing)

            for i in range(20):
                genotype = {"idx": i}

                mapping.update(genotype, i)

                for j in range(len(mapping.genotypes)):
                    self.assertEqual(
                        mapping.genotypes[j]["idx"], max(i - n_storing + 1, 0) + j
                    )
                    self.assertEqual(
                        mapping.fitnesses[j], max(i - n_storing + 1, 0) + j
                    )

        ## Ensure memory safety
        mapping = self._get_mapping()

        original_genotype = {"a": 0, "b": 1}
        genotype = deepcopy(original_genotype)

        mapping.update(genotype, 5)

        self.assertEqual(original_genotype, genotype)

    def test_getitem(self):
        """
        Testing GenotypeMapping.__getitem__.

        Parameters
        ----------
        genotype: genotype
            Genotype to get mapping of.

        Returns
        -------
        fitness if mapped previously else None.
        """
        ## Ensure returns None when not stored
        mapping = self._get_mapping()

        mapping.genotypes = [{"a": 0, "b": 1}, {"b": 1, "c": 2}, {"c": 2, "d": 3}]
        mapping.fitnesses = [3, 2, 1]

        for genotype in [{"a": 1, "b": 1}, {"c": 1, "b": 2}, {"a": 0}]:
            self.assertEqual(mapping[genotype], None)

        ## Ensure finds when correct
        mapping = self._get_mapping()

        mapping.genotypes = [{"a": 0, "b": 1}, {"b": 1, "c": 2}, {"c": 2, "d": 3}]
        mapping.fitnesses = [3, 2, 1]

        for genotype, expected in [({"a": 0, "b": 1}, 3), ({"c": 2, "b": 1}, 2)]:
            self.assertEqual(mapping[genotype], expected)

        ## Test with arbitrary classes
        # -> 2 types of nuerons
        mapping = self._get_mapping()

        FITNESS = 62

        mapping.genotypes = [{"a": DropOff, "b": 1}]
        mapping.fitnesses = [FITNESS]

        # Expected false
        genotype = {"a": LinearDecay, "b": 1}
        self.assertEqual(mapping[genotype], None)

        # Expected true
        genotype = {"a": DropOff, "b": 1}
        self.assertEqual(mapping[genotype], FITNESS)

        ## Ensure memory safety
        mapping = self._get_mapping()

        original_genotype = {"a": 0, "b": 1}
        genotype = deepcopy(original_genotype)

        mapping[genotype]

        self.assertEqual(original_genotype, genotype)


if __name__ == "__main__":
    unittest.main()
