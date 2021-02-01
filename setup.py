"""
Setup script for pip.
"""
from setuptools import setup

setup_args = {
    "name": "spikey",
    "version": "0.0.00",  # x.x.xxxx
    "description": "Spikey is a malleable ndarray based spiking neural network framework and training platform.",
    "license": "MIT",
    "author": "Cole",
    "author_email": "csdhv9@umsystem.edu",
    "packages": ["spikey"],
    "scripts": [],
}


if __name__ == "__main__":
    with open("README.md", "r") as f:
        setup_args["long_description"] = f.read()

    with open("requirements.txt", "r") as f:
        requirements = []

        for value in f.readlines():
            value = value.split("#")[0].strip()

            if not value:
                continue

            requirements.append(value)

        setup_args["install_requires"] = requirements

    setup(**setup_args)
