# Spikey Auto-docs

Spikey's reference is automatically generated from function, class and
module docstrings. docs/ contains hardcoded docs and the generation code.

## Build Documentation

Documentation may be a bit rougher if built in windows, still works well though.

```bash
cd docs/
pip install -r requirements.txt
make docs PYTHON3FUNC=<python_function, default=python3>
```
