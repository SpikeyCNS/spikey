PYTHON3FUNC = python3

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXSOURCEDIR     = .
SPHINXBUILDDIR      = ../_build

all:

docs: 
	@$(PYTHON3FUNC) -m m2r ../README.md --overwrite
	mv ../README.rst README.rst
	sphinx-apidoc -o module ../spikey
	@$(SPHINXBUILD) -M html "$(SPHINXSOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(SPHINXBUILDDIR)"
	rm -rf "$(SPHINXSOURCEDIR)/module"
	rm "$(SPHINXSOURCEDIR)/README.rst"

.PHONY: docs clean
