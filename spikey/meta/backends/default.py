"""
Python multiprocessing distributed backend.
"""
import multiprocessing


class MultiprocessBackend:
    def __init__(self, max_process=16):
        self.max_process = max_process

        self.pool = multiprocessing.Pool(processes=self.max_process)

    def __delete__(self, instance):
        super().__delete__(instance)
        self.pool.close()

    def distribute(self, function: callable, params: list):
        """
        Run function with all sets of parameters given.
        """
        if self.max_process == 1:
            results = []
            for param in params:
                results.append(function(*param))

        else:
            results = self.pool.starmap(function, params)

        return results
