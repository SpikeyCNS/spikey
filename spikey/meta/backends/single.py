"""
Python simple backend.
"""


class SingleProcessBackend:
    def __init__(self):
        pass

    def distribute(self, function: callable, params: list):
        """
        Run function with all sets of parameters given.
        """
        results = []
        for param in params:
            results.append(function(*param))

        return results
