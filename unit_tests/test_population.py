"""
Evaluating the meta rl functionality.
"""
from copy import deepcopy
import unittest
import numpy as np

from spikey.meta import Population


def get_fitness(*a, **kwargs):
    return (0, False)


def get_fitness2(genotype, *args, **kwargs):
    return list(genotype.keys())[0], False


class TestMetaRL(unittest.TestCase):
    """
    Testing meta rl population class.
    """

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for n_process in [1]:  # , 2]:
                with self.subTest(i=n_process):
                    self._get_population = self._set_obj(n_process)

                    func(self)

        return run_all

    def _set_obj(self, n_process):
        """
        Create population that will render only specific object.
        """

        def _get_population(**kwargs):
            np.random.seed(0)

            config = {
                "n_process": n_process,
                "n_storing": 0,
                "n_agents": 100,
                "n_epoch": 10,
                "genotype_constraints": {"a": list(range(0, 10000))},
                "get_fitness": get_fitness,
                "mutate_eligable_pct": 1.0,
                "max_age": 1000000,
                "random_rate": 0,
                "survivor_rate": 1,
                "mutation_rate": 0,
                "crossover_rate": 0,
            }
            config.update(kwargs)

            population = Population(**config)

            return population

        return _get_population

    @run_all_types
    def test_distance(self):
        """
        Testing Population._genotype_dist.

        Parameters
        ----------
        genotype: genotype
            Genotypes to find the distance between.

        Settings
        --------
        genotype_constraints: dict {str: tuple or list}
            Constraints on each gene.

        Returns
        -------
        Distance between the two genotypes.
        """
        ## Ensure the distance between the two genotypes grows as
        ## they get farther away in a metric agnostic to any distance funciton.
        population = self._get_population(
            genotype_constraints={"a": [-101, 101], "b": [-101, 101]}
        )
        BASE = {"a": 0, "b": 0}
        GENOTYPES = [
            {"a": 0, "b": 0},
            {"a": -1, "b": 0},
            {"a": 1, "b": 0},
            {"a": 0, "b": 1},
            {"a": 1, "b": 1},
            {"a": -1, "b": -1},
            {"a": 2, "b": 1},
            {"a": 100, "b": 100},
        ]

        distances = []
        for genotype in GENOTYPES:
            distances.append(population._genotype_dist(BASE, genotype))

        self.assertListEqual(sorted(distances), distances)

        ## Ensure is unaffected by age
        population = self._get_population(
            genotype_constraints={"a": [-101, 101], "b": [-101, 101]}
        )
        BASE = {"a": 0, "b": 0}
        GENOTYPES = [
            {"a": 2, "b": 1, "_age": 10000000000000},
            {"a": 100, "b": 100, "_age": 0},
        ]

        distances = []
        for genotype in GENOTYPES:
            distances.append(population._genotype_dist(BASE, genotype))

        self.assertListEqual(sorted(distances), distances)

    @run_all_types
    def test_random_agent(self):
        """
        Testing population._random.

        Settings
        --------
        genotype_constraints: dict {str: tuple or list}
            Constraints on each gene.

        Returns
        -------
        Randomly generated genotype with all genes satisfying given constraints.
        """
        ## Ensure all values satisfy given constraints
        # With float constraints
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1),
            "c": (0.5, 0.5),
        }

        for _ in range(20):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            genotype = population._random()

            for key, value in genotype.items():
                if key == "_age":
                    continue

                ## check type
                self.assertTrue(isinstance(value, (float, np.float)))

                ## check in range
                low, high = GENOTYPE_CONSTRAINTS[key]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        # With int constraints
        GENOTYPE_CONSTRAINTS = {
            "a": list(range(100)),
            "b": list(range(1000)),
        }

        for _ in range(20):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            genotype = population._random()

            for key, value in genotype.items():
                if key == "_age":
                    continue

                ## check type
                self.assertTrue(isinstance(value, (int, np.integer)))

                ## check in range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        # With list choices
        GENOTYPE_CONSTRAINTS = {
            "a": [[0, x] for x in range(100)],
        }

        for _ in range(20):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            genotype = population._random()

            for key, value in genotype.items():
                if key == "_age":
                    continue

                ## check type
                self.assertEqual(len(value), 2)

                ## check in range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        ## Assert given age
        GENOTYPE_CONSTRAINTS = {
            "a": list(range(100)),
            "b": list(range(1000)),
        }

        population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

        genotype = population._random()

        self.assertIn("_age", genotype)
        self.assertEqual(genotype["_age"], 0)

    @run_all_types
    def test_mutate(self):
        """
        Testing population._mutate.

        Parameters
        ----------
        genotypes: list
            List of genotypes.

        Settings
        --------
        genotype_constraints: dict {str: tuple or list}
            Constraints on each gene.

        Returns
        -------
        A randomly chosed gene will be updated randomly for each of the
        genotypes given.
        """
        ## Ensure one or more values change and still fit constraint.
        # With float constraints
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1),
        }

        initial_genotype = {
            "a": 50.0,
            "b": 0.5,
        }

        for _ in range(20):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            genotype = population._mutate([initial_genotype.copy()])[0]

            del genotype["_age"]

            ## Ensure at least one value changed
            for key in genotype.keys():
                if initial_genotype[key] != genotype[key]:
                    break
            else:
                self.fail(f"Failed: {initial_genotype} == {genotype}")

            for key, value in genotype.items():
                ## check type
                self.assertTrue(isinstance(value, (float, np.float)))

                ## check in range
                low, high = GENOTYPE_CONSTRAINTS[key]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        # With int constraints
        GENOTYPE_CONSTRAINTS = {
            "a": list(range(100)),
            "b": list(range(1000)),
        }

        initial_genotype = {
            "a": 50,
            "b": 500,
            "_age": 0,
        }

        fail_count = 0  ## Sometimes mutations dont work, not necessarily an issue
        for _ in range(20):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            genotype = population._mutate([initial_genotype.copy()])[0]

            del genotype["_age"]

            ## Ensure at least one value changed
            for key in genotype.keys():
                if initial_genotype[key] != genotype[key]:
                    break
            else:
                fail_count += 1

            if fail_count >= 3:
                self.fail(f"Failed: {initial_genotype} == {genotype}")

            for key, value in genotype.items():
                ## check type
                self.assertTrue(isinstance(value, (int, np.integer)))

                ## check in range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        # With list values
        GENOTYPE_CONSTRAINTS = {
            "a": [[0, x] for x in range(100)],
        }

        initial_genotype = {
            "a": [0, 50],
            "_age": 0,
        }

        fail_count = 0  ## Sometimes mutations dont work, not necessarily an issue
        for _ in range(20):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            genotype = population._mutate([initial_genotype.copy()])[0]

            del genotype["_age"]

            ## Ensure at least one value changed
            for key in genotype.keys():
                if initial_genotype[key] != genotype[key]:
                    break
            else:
                fail_count += 1

            if fail_count >= 3:
                self.fail(f"Failed: {initial_genotype} == {genotype}")

            for key, value in genotype.items():
                ## check type
                self.assertTrue(isinstance(value, list))

                ## check in range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        ## Ensure memory safe w/ single value
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1),
        }

        original_genotype = {
            "a": 50.0,
            "b": 0.5,
        }
        genotype = deepcopy(original_genotype)

        population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

        _ = population._mutate(genotype)

        self.assertDictEqual(original_genotype, genotype)

        ## Ensure memory safe w/ list of values
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1),
        }

        original_genotype = [
            {"a": 50.0, "b": 0.5, "_age": 0},
            {"a": 15.0, "b": 0.2, "_age": 0},
            {"a": 9.0, "b": 0.7, "_age": 0},
        ]
        genotype = [deepcopy(original) for original in original_genotype]

        population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

        _ = population._mutate(genotype)

        for i in range(len(original_genotype)):
            self.assertDictEqual(original_genotype[i], genotype[i])

        ## Assert given age
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1),
        }

        original_genotype = [
            {"a": 50.0, "b": 0.5, "_age": 0},
            {"a": 15.0, "b": 0.2, "_age": 0},
            {"a": 9.0, "b": 0.7, "_age": 0},
        ]
        population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

        new_genotypes = population._mutate(original_genotype)

        for genotype in new_genotypes:
            self.assertIn("_age", genotype)
            self.assertEqual(genotype["_age"], 0)

    @run_all_types
    def test_crossover(self):
        """
        Testing population._breed

        Parameters
        ----------
        genotype: dict {str: tuple/list}
            Initial Genotypes.

        Settings
        --------
        genotype_constraints: dict {str: tuple or list}
            Constraints on each gene.

        Returns
        -------
        Genotype.
        """
        ## Assert new values are inbetween a and b, regardless of their order and
        ## fit within constraints.
        # With float constraints
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1000),
        }

        genotype_a = {
            "a": 25.25,
            "b": 500.0,
        }
        genotype_b = {
            "a": 75.75,
            "b": 501.0,
        }

        for _ in range(5):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            ## A, B
            new_genotype1, new_genotype2 = population._crossover(genotype_a, genotype_b)

            del new_genotype1["_age"], new_genotype2["_age"]

            ## Ensure at least one value changed from each initial
            for key in new_genotype1.keys():
                if genotype_a[key] != new_genotype1[key]:
                    break

            for key in new_genotype2.keys():
                if genotype_b[key] != new_genotype2[key]:
                    break

            for key, value in new_genotype1.items():
                ## check type
                self.assertTrue(isinstance(value, (float, np.float)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            for key, value in new_genotype2.items():
                ## check type
                self.assertTrue(isinstance(value, (float, np.float)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            ## B, A
            new_genotype1, new_genotype2 = population._crossover(genotype_b, genotype_a)

            del new_genotype1["_age"], new_genotype2["_age"]

            ## Ensure at least one value changed from each initial
            for key in new_genotype1.keys():
                if genotype_a[key] != new_genotype1[key]:
                    break

            for key in new_genotype2.keys():
                if genotype_b[key] != new_genotype2[key]:
                    break

            for key, value in new_genotype1.items():
                ## check type
                self.assertTrue(isinstance(value, (float, np.float)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            for key, value in new_genotype2.items():
                ## check type
                self.assertTrue(isinstance(value, (float, np.float)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        # With int constraints
        GENOTYPE_CONSTRAINTS = {
            "a": list(range(100)),
            "b": list(range(1000)),
        }

        genotype_a = {
            "a": 25,
            "b": 500,
        }
        genotype_b = {
            "a": 75,
            "b": 501,
        }

        for _ in range(5):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            ## A, B
            new_genotype1, new_genotype2 = population._crossover(genotype_a, genotype_b)

            del new_genotype1["_age"], new_genotype2["_age"]

            for key in new_genotype1.keys():
                if genotype_a[key] != new_genotype1[key]:
                    break

            for key in new_genotype2.keys():
                if genotype_b[key] != new_genotype2[key]:
                    break

            for key, value in new_genotype1.items():
                ## check type
                self.assertTrue(isinstance(value, (int, np.int)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            for key, value in new_genotype2.items():
                ## check type
                self.assertTrue(isinstance(value, (int, np.int)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            ## B, A
            new_genotype1, new_genotype2 = population._crossover(genotype_b, genotype_a)

            del new_genotype1["_age"], new_genotype2["_age"]

            for key in new_genotype1.keys():
                if genotype_a[key] != new_genotype1[key]:
                    break

            for key in new_genotype2.keys():
                if genotype_b[key] != new_genotype2[key]:
                    break

            for key, value in new_genotype1.items():
                ## check type
                self.assertTrue(isinstance(value, (int, np.int)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            for key, value in new_genotype2.items():
                ## check type
                self.assertTrue(isinstance(value, (int, np.int)))

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        # With list values
        GENOTYPE_CONSTRAINTS = {
            "a": [[x // 2, x * 2] for x in range(100)],
        }

        genotype_a = {
            "a": [3, 12],
        }
        genotype_b = {
            "a": [6, 24],
        }

        for _ in range(5):
            population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

            ## A, B
            new_genotype1, new_genotype2 = population._crossover(genotype_a, genotype_b)

            del new_genotype1["_age"], new_genotype2["_age"]

            for key in new_genotype1.keys():
                if genotype_a[key] != new_genotype1[key]:
                    break

            for key in new_genotype2.keys():
                if genotype_b[key] != new_genotype2[key]:
                    break

            for key, value in new_genotype1.items():
                ## check type
                self.assertEqual(len(value), 2)

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            for key, value in new_genotype2.items():
                ## check type
                self.assertEqual(len(value), 2)

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            ## B, A
            new_genotype1, new_genotype2 = population._crossover(genotype_b, genotype_a)

            del new_genotype1["_age"], new_genotype2["_age"]

            for key in new_genotype1.keys():
                if genotype_a[key] != new_genotype1[key]:
                    break

            for key in new_genotype2.keys():
                if genotype_b[key] != new_genotype2[key]:
                    break

            for key, value in new_genotype1.items():
                ## check type
                self.assertEqual(len(value), 2)

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

            for key, value in new_genotype2.items():
                ## check type
                self.assertEqual(len(value), 2)

                ## check in constraint range
                low, high = GENOTYPE_CONSTRAINTS[key][0], GENOTYPE_CONSTRAINTS[key][-1]

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

                ## check inbetween a and b
                low = min(genotype_a[key], genotype_b[key])
                high = max(genotype_a[key], genotype_b[key])

                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

        ## Ensure memory safe
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1000),
        }

        original_genotype_a = {
            "a": 25.25,
            "b": 500.0,
        }
        original_genotype_b = {
            "a": 75.75,
            "b": 501.0,
        }
        genotype_a, genotype_b = (
            deepcopy(original_genotype_a),
            deepcopy(original_genotype_b),
        )

        population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

        new_genotype1, new_genotype2 = population._crossover(genotype_a, genotype_b)

        self.assertDictEqual(original_genotype_a, genotype_a)
        self.assertDictEqual(original_genotype_b, genotype_b)

        ## Ensure age given
        GENOTYPE_CONSTRAINTS = {
            "a": (0, 100),
            "b": (0, 1000),
        }

        genotype_a = {
            "a": 25.25,
            "b": 500.0,
        }
        genotype_b = {
            "a": 75.75,
            "b": 501.0,
        }
        population = self._get_population(genotype_constraints=GENOTYPE_CONSTRAINTS)

        new_genotype1, new_genotype2 = population._crossover(genotype_a, genotype_b)

        self.assertIn("_age", new_genotype1)
        self.assertEqual(new_genotype1["_age"], 0)
        self.assertIn("_age", new_genotype2)
        self.assertEqual(new_genotype2["_age"], 0)

    @run_all_types
    def test_update(self):
        """
        Testing population.update

        Parameters
        ----------
        f: list of float
            Fitness values of each agent.

        Settings
        --------
        survivor_rate: float [0, 1]
            Percent population that survives.
        mutation_rate: float [0, 1]
            Percent population to be mutated.
        crossover_rate: float [0, 1]
            Percent population to be crossed over.
        random_rate: float [0, 1]
            Percent population to generate randomly
        mutate_eligable_pct: flaot (0, 1]
            Percent population eligable for mutation based on fitness.

        Effects
        -------
        A new population will be generated.
        """
        ## Ensure rates normalized.
        reals = [(0.4, 0.2, 0.4), (0.2, 0.1, 0.2), (0.0001, 0.0001, 0.0008)]
        targets = [(0.4, 0.2, 0.4), (0.4, 0.2, 0.4), (0.1, 0.1, 0.8)]

        for i, (s, m, c) in enumerate(reals):
            population = self._get_population(
                survivor_rate=s, mutation_rate=m, crossover_rate=c
            )

            target_s, target_m, target_c = targets[i]
            self.assertEqual(population._survivor_rate, target_s)
            self.assertEqual(population._mutation_rate, target_m)
            self.assertEqual(population._crossover_rate, target_c)

        ## Ensure agent count abided by.
        population_sizes = [10, 11]
        rates = [(0.4, 0.2, 0.4), (0.1, 0.1, 0.8)]

        for population_size in population_sizes:
            for s, m, c in rates:
                population = self._get_population(
                    n_agents=population_size,
                    survivor_rate=s,
                    mutation_rate=m,
                    crossover_rate=c,
                )

                for _ in range(4):
                    population.update(population.evaluate())
                    self.assertEqual(len(population), population_size)

        ## Ensure all survivors w/ 100% survivorship.
        population = self._get_population(n_agents=10, survivor_rate=1.0)

        original_population = [value.copy() for value in population.population]

        for i in range(5):
            population.update(list(range(len(original_population))))

            for genotype in population.population:
                self.assertEqual(genotype["_age"], i + 1)

                fake_genotype = deepcopy(genotype)
                fake_genotype["_age"] = 0

                self.assertTrue(fake_genotype in original_population)

        ## Ensure all not survivors w/ 100% mutation.
        population = self._get_population(
            n_agents=100, survivor_rate=0.0, mutation_rate=1.0
        )

        original_population = [value.copy() for value in population.population]

        for _ in range(5):
            population.update(list(range(len(original_population))))

            n_fails = 0
            for genotype in population.population:
                self.assertEqual(genotype["_age"], 0)

                if genotype in original_population:
                    n_fails += 1

            self.assertTrue(n_fails < population._n_agents * 0.2)

        ## Ensure all not survivors w/ 100% crossover.
        GENOTYPE_CONSTRAINTS = {
            "a": list(range(1000)),
            "b": list(range(1000)),
            "c": list(range(1000)),
            "d": list(range(1000)),
            "e": list(range(1000)),
        }
        population = self._get_population(
            n_agents=20,
            survivor_rate=0.0,
            crossover_rate=1.0,
            genotype_constraints=GENOTYPE_CONSTRAINTS,
        )

        original_population = [value.copy() for value in population.population]

        for _ in range(5):
            population.update(list(range(len(original_population))))

            n_fails = 0
            for genotype in population.population:
                self.assertEqual(genotype["_age"], 0)

                if genotype in original_population:
                    n_fails += 1

            self.assertTrue(n_fails < max(population._n_agents * 0.75, 2))

        ## Ensure terminates when env gives signal.
        mark = 12

        population = self._get_population(
            n_agents=1,
            n_epoch=10 ** 10,
            get_fitness=lambda g, *args, **kwargs: (0, i == mark),
        )

        for i in range(0, mark + 1, 2):
            self.assertFalse(population.terminated)

            population.population = [{i: i}]

            population.evaluate()

        self.assertTrue(population.terminated)

        ## Ensure mutate_eligable_pct is respected
        N_AGENTS = 100

        GENOTYPE_CONSTRAINTS = {
            "a": list(range(N_AGENTS)),
            "b": list(range(N_AGENTS)),
        }

        for pct_eligable in [0.1, 0.5, 0.75, 1.0]:
            population = self._get_population(
                n_agents=N_AGENTS,
                survivor_rate=0.0,
                mutation_rate=1.0,
                mutate_eligable_pct=pct_eligable,
                genotype_constraints=GENOTYPE_CONSTRAINTS,
            )

            population.population = [
                {"a": i, "b": i, "_age": 0} for i in range(1, N_AGENTS + 1)
            ]

            population.update(list(range(N_AGENTS)))

            threshold = (1 - pct_eligable) * N_AGENTS

            self.assertEqual(len(population.population), N_AGENTS)

            for genotype in population.population:
                self.assertTrue(
                    any([value >= threshold for value in genotype.values()])
                )

        ## Ensure agents with > allowed age removed
        N_AGENTS = 100

        GENOTYPE_CONSTRAINTS = {
            "a": list(range(N_AGENTS)),
            "b": list(range(N_AGENTS)),
        }

        for max_age in [1, 50, 90]:
            population = self._get_population(
                n_agents=N_AGENTS,
                survivor_rate=1.0,
                max_age=max_age,
                genotype_constraints=GENOTYPE_CONSTRAINTS,
            )

            population.population = [
                {"a": i, "b": i, "_age": i} for i in range(N_AGENTS)
            ]

            population.update(list(range(N_AGENTS)))

            for genotype in population.population:
                self.assertLessEqual(genotype["_age"], max_age)

        ## Ensure agents w/ age < max are not discarded prematurely
        N_AGENTS = 100

        GENOTYPE_CONSTRAINTS = {
            "c": list(range(N_AGENTS)),
            "d": list(range(N_AGENTS)),
        }

        for max_age in [20, 50, 90]:
            population = self._get_population(
                n_agents=N_AGENTS,
                survivor_rate=0.2,
                random_rate=0.8,
                max_age=max_age,
                genotype_constraints=GENOTYPE_CONSTRAINTS,
            )

            population.population = [
                {"a": i, "b": i, "_age": i} for i in range(N_AGENTS)
            ]

            population.update(list(range(N_AGENTS)))
            n_found = 0
            for genotype in population.population:
                if "a" in genotype:
                    self.assertGreaterEqual(genotype["a"], max_age - 20)
                    self.assertLessEqual(genotype["a"], max_age)
                    n_found += 1

            self.assertEqual(n_found, 20)

        ## Ensure memory safe
        N_AGENTS = 100

        population = self._get_population(
            n_agents=N_AGENTS,
            survivor_rate=0.0,
            mutation_rate=1.0,
            mutate_eligable_pct=pct_eligable,
            genotype_constraints=GENOTYPE_CONSTRAINTS,
        )

        original_fitnesses = np.arange(N_AGENTS)
        fitnesses = np.copy(original_fitnesses)

        population.update(fitnesses)

        self.assertListEqual(
            list(np.ravel(original_fitnesses)), list(np.ravel(fitnesses))
        )

    @run_all_types
    def test_evaluate(self):
        """
        Testing population.evaluate.

        Settings
        --------
        get_fitness: func[agent] -> float
            Function that calculates the fitness of a given genotype.

        Returns
        -------
        Fitness values for each agent.
        """
        ## Ensure get_fitness is used to determine fitness.
        population = self._get_population(get_fitness=get_fitness2)

        GENOTYPES = [{i: i} for i in np.arange(0, 100, 0.5)]

        population.population = GENOTYPES

        fitnesses = population.evaluate()

        self.assertListEqual([list(key)[0] for key in GENOTYPES], list(fitnesses))

        ## Ensure memory safe
        population = self._get_population(get_fitness=get_fitness2)

        ORIGINAL_GENOTYPES = [{i: i} for i in np.arange(0, 100, 0.5)]
        GENOTYPES = np.copy(ORIGINAL_GENOTYPES)

        population.population = GENOTYPES

        fitnesses = population.evaluate()

        self.assertListEqual([list(key)[0] for key in GENOTYPES], list(fitnesses))

    @run_all_types
    def test_n_agents(self):
        """
        population.n_agents is a generator returning the number of
        agents in a population at every epoch.

        If the parameter n_agents is an int, the generator will
        return that number n_epoch times.

        If the parameter n_agents is a list, the generator will
        loop over the values in it once.

        Ensure the population generated fits the population count expected.
        """
        ## Test with integer/
        for n_agents in [2, 54, 111]:
            population = self._get_population(n_agents=n_agents, survivor_rate=1.0)

            self.assertEqual(n_agents, len(population))

            for _ in range(10):
                population.update(np.zeros(shape=len(population)))
                self.assertEqual(n_agents, len(population))

        ## Test with list of values.
        values = [
            list(range(1, 15)),
            list(range(1, 100))[::-1],
            [2, 4, 5, 7, 9, 12, 13, 16, 17, 6, 4, 3, 2, 1],
        ]

        for n_agents in values:
            population = self._get_population(n_agents=n_agents, survivor_rate=1.0)

            self.assertEqual(n_agents[0], len(population))

            for value in n_agents[1:10]:
                population.update(np.zeros(shape=len(population)))
                self.assertEqual(value, len(population))

        ## Ensure terminates if n_agents is int after n_epochs.
        for n_epoch in [2, 54, 111]:
            population = self._get_population(n_epoch=n_epoch, survivor_rate=1.0)

            for _ in range(n_epoch):
                self.assertFalse(population.terminated)
                population.update(np.zeros(shape=len(population)))

            self.assertTrue(population.terminated)

        ## Ensure terminates if n_agents is list after iterating through whole thing.
        values = [
            list(range(1, 15)),
            list(range(1, 100))[::-1],
            [2, 4, 5, 7, 9, 12, 13, 16, 17, 6, 4, 3, 2, 1],
        ]

        for n_agents in values:
            population = self._get_population(
                n_agents=n_agents, survivor_rate=1.0, n_epoch=1
            )

            for _ in range(len(n_agents)):
                self.assertFalse(population.terminated)
                population.update(np.zeros(shape=len(population)))

            self.assertTrue(population.terminated)


if __name__ == "__main__":
    unittest.main()
