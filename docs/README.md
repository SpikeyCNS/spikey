# Spikey Auto-docs

Spikey's reference documentation is automatically generated from function, class and
module docstrings using Sphinx.

## Build Documentation

NOTE: Documentation may be a bit rough if built in windows.

```bash
cd docs/
pip install -r requirements.txt
make docs PYTHON3FUNC=<python_function, default=python3>
```
