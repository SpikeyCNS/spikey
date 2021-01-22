"""
Template runtime benchmark.
"""
import common


class TimeTemplate:
    """
    Timing module.
    """

    def setup(self):
        """
        Load parameters.
        """
        ##
        self.PARAMETERS = {}  # {name: (arg1, arg2, ...)}

    def time_demo(self, *args):
        """
        Function to time here, may create multiple.
        """
