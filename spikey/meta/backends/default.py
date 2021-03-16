"""
Distributed backend using multiprocessing pool.
"""
import multiprocessing
from spikey.meta.backends.template import MetaBackend


class MultiprocessBackend(MetaBackend):
    """
    Distributed backend using multiprocessing pool.
    """

    def __init__(self, max_process: int = 16):
        self.max_process = max_process

        if self.max_process > 1:
            self.pool = multiprocessing.Pool(processes=self.max_process)

    def __delete__(self, instance: object):
        self.pool.close()
        super().__delete__(instance)

    def distribute(self, function: callable, params: list) -> list:
        """
        Execute function on each set of parameters.

        Parameters
        ----------
        function: callable(\*param) -> any
            Function to execute.
        params: list
            List of different params to execute function on.

        Returns
        -------
        list Return value of function for each set of parameters given.
        """
        if self.max_process == 1:
            results = []
            for param in params:
                results.append(function(*param))

        else:
            results = self.pool.starmap(function, params)

        return results
