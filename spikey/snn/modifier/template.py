"""
A variable modifier.
"""


class Modifier:
    """
    A network variable modifier.
    """

    def __init__(self, param: list, *_):
        self.param = param

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError(f"__eq__ not implmeneted for {type(self)}!")

    def set_param(self, network: "SNN", value: float):
        """
        Update given parameter to value
        """
        if self.param[0] == "network":
            obj = network

            for item in self.param[1:-1]:
                obj = obj.__dict__[item]

            obj.__dict__[self.param[-1]] = value
        else:
            raise ValueError(f"Param not recognized {self.param[0]}!")

    def update(self, network: "SNN"):
        raise NotImplementedError(f"Update not implemented for {type(self)}!")
