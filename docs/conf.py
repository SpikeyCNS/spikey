# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.

# -- Project information -----------------------------------------------------

from setup import setup_args

project = setup_args["name"]
author = setup_args["author"]
release = setup_args["version"]
copyright = "2021, Spikey"


# -- General configuration ---------------------------------------------------

master_doc = "index"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_static_path = []
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "external_links": [
      {"name": "Reference", "url": "py-modindex.html"},
    ],
    "show_prev_next": False,
}
