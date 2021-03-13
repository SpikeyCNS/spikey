"""
Tests for snn.Network.
"""
import unittest
from unit_tests import ModuleTest
from spikey.meta.population import GenotypeMapping, Population


class TestGenotypeMapping(unittest.TestCase, ModuleTest):
    """
    Tests for meta.GenotypeMapping.
    """

    TYPES = [GenotypeMapping]
    BASE_CONFIG = {"n_storing": 10}

    @ModuleTest.run_all_types
    def test_usage(self):
        n_storing = self.BASE_CONFIG['n_storing']
        cache = self.get_obj()

        for fitness_original, genotype in enumerate([{i: i} for i in range(n_storing*10)]):
            cache.update(genotype, fitness_original)
            fitness = cache[genotype]
            self.assertEqual(fitness, fitness_original)
            fitness = cache[{'not_in': 2}]
            self.assertEqual(fitness, None)

        self.assertEqual(len(cache.genotypes), self.BASE_CONFIG['n_storing'])


if __name__ == "__main__":
    unittest.main()
