"""
benchmarks/runtime __init__.
"""
import sys
import os


try:
    modules = []

    for filename in os.listdir(os.path.join("benchmarks", "runtime")):
        if filename is "runall.py":
            continue

        if filename[:5] != "bench" or filename[-3:] != ".py":
            continue

        modules.append(__import__(f"{filename[:-3]}"))

except ImportError as e:
    raise ImportError(f"benchmarks/runtime/__init__.py failed: {e}")
