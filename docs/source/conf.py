# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import mock

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "coolpup.py"
copyright = "2020, Ilya M. Flyamer"
author = "Ilya M. Flyamer"

# The full version, including alpha/beta/rc tags
import re

VERSIONFILE = "../../coolpuppy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
release = verstr


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    # 'sphinx_argparse_cli',
    # 'myst_parser',
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.githubpages",
    # "sphinx.ext.mathjax",
    "sphinxarg.ext",
    # 'm2r2'
]
napoleon_numpy_docstring = True

MOCK_MODULES = [
    "Cython",
    "bioframe",
    "cooler",
    "cooltools",
    "cooltools.lib",
    "cooltools.api",
    "h5py",
    "h5sparse",
    "matplotlib",
    "matplotlib.colors",
    "matplotlib.font_manager",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "matplotlib.gridspec",
    "more_itertools",
    "mpl_toolkits.axes_grid1",
    "natsort",
    "numpy",
    "pandas",
    "pysam",
    "scipy",
    "scipy.linalg",
    "seaborn",
    "multiprocessing_logging",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

autodoc_mock_imports = MOCK_MODULES

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

add_module_names = True

# master_doc = 'index'


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.add_css_file("my_theme.css")


source_suffix = [".rst", ".md", ".ipynb"]
html_favicon = "favicon.ico"

autodoc_docstring_signature = True
nbsphinx_execute = "never"
jupyter_execute_notebooks = "off"
