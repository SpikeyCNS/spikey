"""
Update links.
"""
old_link = "https://github.com/SpikeyCNS/spikey"
new_link = "https://spikeycns.github.io/index.html"

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]

    with open(filename, 'r') as file:
        data = file.read()

    data = data.replace(old_link, new_link)

    with open(filename, 'w') as file:
        file.write(data)
