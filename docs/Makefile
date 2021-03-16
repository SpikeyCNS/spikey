PYTHON3FUNC = python3

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXSOURCEDIR     = .
SPHINXBUILDDIR      = ../_build

all: docs

docs:
	@$(PYTHON3FUNC) -m m2r ../README.md --overwrite
	mv ../README.rst index.rst
	sphinx-apidoc -o module ../spikey -t _templates/
	@$(SPHINXBUILD) -M html "$(SPHINXSOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(SPHINXBUILDDIR)"
	rm -rf "$(SPHINXSOURCEDIR)/module"
	rm "$(SPHINXSOURCEDIR)/index.rst"

.PHONY: docs clean
