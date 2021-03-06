"""
Base unit test modules for spikey.
"""
import numpy as np
from copy import deepcopy
import pickle

np.random.seed(0)


class BaseTest:
    """
     Spikey's base unit test case. Any methods defined here
     will be run on all tests that inheret this, unless overriden.

     Usage
     -----
     ```python
     class TestCustom(unittest.TestCase, BaseTest):
         TYPES = [object types to test]
         BASE_CONFIG = {...}

         @BaseTest.run_all_types
         def test_update(self):
             obj = self.get_obj({config updates})
    ```
    """

    TYPES = []  # List of object types to test
    BASE_CONFIG = {}  # Base object configuration

    def _set_obj(self, obj):
        """
        Create generator that will render only specific object type.
        """

        def get_obj(**kwargs):
            base_config = deepcopy(self.BASE_CONFIG)
            base_config.update(kwargs)
            generator = obj(**base_config)
            return generator

        return get_obj

    def run_all_types(func):
        """
        Wrapper creating subtest for every type of object.
        """

        def run_all(self):
            for obj in self.TYPES:
                with self.subTest(i=obj.__name__):
                    self.get_obj = self._set_obj(obj)
                    func(self)

        return run_all


class ModuleTest(BaseTest):
    """
    Spikey's base unit test case. Any methods defined here
    will be run on all tests that inheret this, unless overriden.

    Usage
    -----
    ```python
    class TestCustom(unittest.TestCase, ModuleTest):
        TYPES = [object types to test]
        BASE_CONFIG = {...}

        @ModuleTest.run_all_types
        def test_update(self):
            obj = self.get_obj({config updates})
    ```
    """

    @BaseTest.run_all_types
    def test_modulecopy(self):
        """
        Ensure module.copy() works well on obj.
        """
        a = self.get_obj()
        b = a.copy()

        self.assertIsInstance(b, type(a))
        for key, value in a.__dict__.items():
            with self.subTest(key):
                self.assertTrue(hasattr(b, key))
                self.assertIsInstance(getattr(b, key), type(value))

    @BaseTest.run_all_types
    def test_pickle(self):
        """
        Ensure pickle works well on obj.
        """
        a = self.get_obj()
        b = pickle.loads(pickle.dumps(a))

        self.assertIsInstance(b, type(a))
        for key, value in a.__dict__.items():
            with self.subTest(key):
                self.assertTrue(hasattr(b, key))
                self.assertIsInstance(getattr(b, key), type(value))
