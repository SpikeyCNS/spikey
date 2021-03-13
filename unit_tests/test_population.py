"""
Tests for snn.Network.
"""
import unittest
from unit_tests import ModuleTest
import os
from spikey.meta.population import GenotypeMapping, Population, checkpoint_population, read_population


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


class FakeMetaRL:
    GENOTYPE_CONSTRAINTS = {
        "a": list(range(8)),
    }
    params = {}

    def get_fitness(self, genotype):
        return list(range(len(genotype)))

class TestPopulation(unittest.TestCase, ModuleTest):
    """
    Tests for meta.Population.
    """

    TYPES = [Population]
    BASE_CONFIG = {
        "game": FakeMetaRL(),
        "logging": False,
        "max_process": 1,
        "n_storing": 256,
        "n_agents": 4,
        "n_epoch": 100,
        "mutate_eligable_pct": 0.5,
        "max_age": 5,
        "random_rate": 0.1,
        "survivor_rate": 0.1,
        "mutation_rate": 0.3,
        "crossover_rate": 0.5,
    }
    FOLDER = "testing_temp_dir"

    @classmethod
    def tearDownClass(cls):
        try:
            for filename in os.listdir(TestPopulation.FOLDER):
                os.remove(os.path.join(TestPopulation.FOLDER, filename))

            os.removedirs(TestPopulation.FOLDER)
        except FileNotFoundError:
            pass

    @ModuleTest.run_all_types
    def test_usage(self):
        population = self.get_obj()
        for _ in range(10):
            fitness = population.evaluate()
            population.update(fitness)

    @ModuleTest.run_all_types
    def test_checkpoint(self):
        population_old = self.get_obj()
        checkpoint_population(population_old, self.FOLDER)

        population_new = self.get_obj()
        read_population(population_new, self.FOLDER)

        self.assertEqual(population_old.population, population_new.population)


if __name__ == "__main__":
    unittest.main()
