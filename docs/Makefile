PYTHON3FUNC = python3

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXSOURCEDIR     = .
SPHINXBUILDDIR      = ../_build

all: docs

docs:
	@$(PYTHON3FUNC) link_fix.py ../README.md index.md
	@$(PYTHON3FUNC) -m m2r index.md --overwrite --parse-relative-links
	export SPHINX_APIDOC_OPTIONS=members,show-inheritance  # Note copy env var to doc build script
	sphinx-apidoc -t _templates/ -o module ../spikey
	@$(SPHINXBUILD) -M html "$(SPHINXSOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(SPHINXBUILDDIR)"
	rm -rf "$(SPHINXSOURCEDIR)/module"
	rm -f "$(SPHINXSOURCEDIR)/index.rst"
	rm -f "$(SPHINXSOURCEDIR)/index.md"

.PHONY: docs clean
