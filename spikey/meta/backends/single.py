"""
Single thread backend.
"""


class SingleProcessBackend:
    """
    Single thread backend.
    """

    def __init__(self):
        pass

    def distribute(self, function: callable, params: list) -> list:
        """
        Execute function on each set of parameters.

        Parameters
        ----------
        function: callable(*param) -> any
            Function to execute.
        params: list
            List of different params to execute function on.

        Returns
        -------
        list Return value of function for each set of parameters given.
        """
        results = []
        for param in params:
            results.append(function(*param))

        return results
