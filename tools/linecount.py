"""
Count the number of lines in .py files in this repo.
"""
import os
import json


def get_filenames(folder):
    """
    Find .py files recursively.

    Parameters
    ----------
    folder: str
        Folder to search.

    Returns
    -------
    list[str] List of filenames.
    """
    filenames = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if os.path.isdir(path) and filename[0] != ".":
            filenames += get_filenames(path)
        elif filename[-3:] == ".py" or filename[-6:] == ".ipynb":
            filenames.append(path)

    return filenames


def count_lines(filename):
    """
    Count the number of lines in a file.

    Parameters
    ----------
    filename: str
        Filename to cound lines of.

    Returns
    -------
    int Number of lines in file.
    """
    try:
        with open(filename, "r") as file:
            if filename[-3:] == ".py":
                return len(file.readlines())
            elif filename[-6:] == ".ipynb":
                try:
                    cells = json.load(file)

                    cells = cells["cells"]

                    return sum(
                        len(c["source"]) for c in cells if c["cell_type"] == "code"
                    )
                except JSONDecodeError:
                    print(f"Cannot read '{filename}' because it is open already!")

            else:
                raise ValueError(f"Unrecognized file type - '{filename}'!")
    except FileNotFoundError:
        pass

    return 0


if __name__ == "__main__":
    filenames = get_filenames(".")

    print(filenames)

    count = sum([count_lines(filename) for filename in filenames])

    print(f"{count} total lines.")
