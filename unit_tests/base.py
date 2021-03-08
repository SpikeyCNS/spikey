"""
Base unit test modules for spikey.
"""


class BaseTest:
    """
    Spikey's base unit test case. Any methods defined here
    will be run on all tests that inheret this, unless overriden.

    Usage
    -----
    ```python
    class TestCustom(unittest.TestCase, BaseTest):
        pass
    ```
    """


class ModuleTest(BaseTest):
    """
    Spikey's base unit test case. Any methods defined here
    will be run on all tests that inheret this, unless overriden.

    Usage
    -----
    ```python
    class TestCustom(unittest.TestCase, ModuleTest):
        pass
    ```
    """
