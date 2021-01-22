"""
Serializing ndarrays for json logging.
"""
import numpy as np


def compressnd(matrix: np.ndarray, precision: int = None) -> str:
    """
    Recursively compress n dimensional ndarray.

    --> ie turn into single line to save space.
    """
    if isinstance(matrix, (list, tuple)):
        matrix = np.array(matrix)

    if len(matrix.shape) > 1:
        return (
            "["
            + ",".join([compressnd(row, precision=precision) for row in matrix])
            + "]"
        )

    if precision:
        return "[" + " ".join([f"{value:.{precision}f}" for value in matrix]) + "]"

    return "[" + " ".join([f"{value}" for value in matrix]) + "]"


def uncompressnd(string: str, n: int = 0) -> np.ndarray:
    """
    Recursively uncompress n dimensional ndarray.

    --> Single line of text to n dimensional ndarray.
    """
    if not string:
        return None

    def create_deep(m):
        if m == 1:
            return []

        return [create_deep(m - 1)]

    def deep_index(matrix, m):
        if m <= 0:
            return matrix
        if m == 1:
            return matrix[-1]

        return deep_index(matrix[-1], m - 1)

    if string[0] == "[":
        return uncompressnd(string[1:-1], n + 1)

    string = string.replace("[", "/[/").replace("]", "/]/").split("/")

    output = create_deep(n)

    depth = n
    for item in string:
        if item == "[":
            depth += 1
        elif item == "]":
            depth -= 1
        elif item == ",":
            deep_index(output, depth - 1).append([])
        elif item == "":
            pass
        else:
            try:
                deep_index(output, depth - 1)
            except IndexError:
                curr_depth = depth

                while curr_depth >= 0 and curr_depth <= depth:
                    try:
                        deep_index(output, curr_depth - 1).append([])
                        break
                    except IndexError:
                        curr_depth -= 1

            for value in item.split(" "):
                deep_index(output, depth - 1).append(float(value))

    return np.array(output)
