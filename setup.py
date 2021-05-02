"""
Setup script for pip.
"""
import sys
from setuptools import setup

assert (
    sys.version_info.major == 3 and sys.version_info.minor >= 6
), "Python version must be >=3.6 to install Spikey!"

setup_args = {
    "name": "spikey",
    "version": "0.5.00",
    "description": "Spikey is a malleable spiking neural network framework and training platform.",
    "license": "MIT",
    "author": "Cole",
    "author_email": "csdhv9@umsystem.edu",
    "packages": ["spikey"],
    "install_requires": ["numpy", "pandas", "matplotlib"],
    "python_requires": ">=3.6",
    "url": "https://github.com/SpikeyCNS/spikey",
    "classifiers": [
        "Programming Language :: Python :: 3.8",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
}


if __name__ == "__main__":
    with open("README.md", "r") as f:
        setup_args["long_description"] = f.read()
        setup_args["long_description_content_type"] = "text/markdown"

    setup(**setup_args)
