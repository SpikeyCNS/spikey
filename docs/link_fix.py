"""
Update links.
"""
if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "Table of Contents" in line:
            a = i
        elif "## Spiking Neural Networks" in line:
            b = i
        if line.startswith("## "):
            key = line[3:].strip().replace(" ", "-").lower()
            lines[i] = f".. _{key}:\n{line.strip()}\n"

    lines = lines[:a] + lines[b:]

    with open(sys.argv[2], "w") as file:
        for line in lines:
            file.write(line)
