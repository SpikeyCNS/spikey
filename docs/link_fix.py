"""
Update links.
"""
if __name__ == "__main__":
    import sys

    filename = sys.argv[1]

    with open(filename, "r") as file:
        data = file.read()

    a = data.index("Table of Contents")
    b = data.index("Spiking Neural Networks\n----")
    data = data[:a] + data[b:]

    with open(filename, "w") as file:
        file.write(data)
