"""
Ndarray serialize functionality for json logging.
"""
import numpy as np


BOOL_MAP = {'True': 1, 'False': 0}

def compressnd(matrix: np.ndarray, precision: int = None) -> str:
    """
    Recursively compress n dimensional ndarray into single line string.
    NOTE: Output string is much more memory inefficient than numpy so
        RAM may overflow with huge matricies.

    Parameters
    ----------
    matrix: np.ndarray
        Array to compress.
    precision: int
        Number of decimals in floating point values.

    Returns
    -------
    str Matrix formatted into single line string.
    "[[1 2 3],[4 5 6]]"

    Usage
    -----
    ```python
    value = compressnd(np.ones(3))
    ```
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


def uncompressnd(string: str, _depth=0) -> np.ndarray:
    """
    Recursively uncompress n dimensional ndarray from single line string.
    Inverse operator to compressnd.

    Parameters
    ----------
    string: str
        String to decompress.
    _depth: int, default=0
        Current depth of recursion, should always be 0 for normal usage.

    Returns
    -------
    np.ndarray Ndarray version of string given.

    Usage
    -----
    ```python
    matrix = uncompressnd("[[1 2 3],[4 5 6]]")
    ```
    """
    if not string:
        return None

    def create_deep(m):
        if m <= 1:
            return []

        return [create_deep(m - 1)]

    def deep_index(matrix, m):
        if m <= 0:
            return matrix
        if m == 1:
            return matrix[-1]

        return deep_index(matrix[-1], m - 1)

    if string[0] == "[":
        return uncompressnd(string[1:-1], _depth + 1)

    string = string.replace("[", "/[/").replace("]", "/]/").split("/")

    output = create_deep(_depth)

    for item in string:
        if item == "[":
            _depth += 1
        elif item == "]":
            _depth -= 1
        elif item == ",":
            deep_index(output, _depth - 1).append([])
        elif item == "":
            pass
        else:
            try:
                deep_index(output, _depth - 1)
            except IndexError:
                curr_depth = _depth

                while curr_depth >= 0 and curr_depth <= _depth:
                    try:
                        deep_index(output, curr_depth - 1).append([])
                        break
                    except IndexError:
                        curr_depth -= 1

            for value in item.split(" "):
                try:
                    value = float(value)
                except ValueError:
                    value = BOOL_MAP[value]

                deep_index(output, _depth - 1).append(value)

    return np.array(output)
