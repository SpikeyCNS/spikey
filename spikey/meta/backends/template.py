"""
Backend for executing a function on each set of parameters given.
Primarily meant for meta analysis tools.
"""


class MetaBackend:
    """
    Backend for executing a function on each set of parameters given.
    Primarily meant for meta analysis tools.
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
        raise NotImplementedError(
            f"ERROR: distribute not implemented for {type(self)}!"
        )
