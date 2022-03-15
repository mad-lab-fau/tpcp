# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import sys
from datetime import datetime
from pathlib import Path

import toml
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

HERE = Path(__file__)

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

from sphinxext.githublink import make_linkcode_resolve

URL = "https://github.com/mad-lab-fau/tpcp"

# -- Project information -----------------------------------------------------

# Info from poetry config:
info = toml.load("../pyproject.toml")["tool"]["poetry"]

project = info["name"]
author = ", ".join(info["authors"])
release = info["version"]

copyright = "2021 - {}, MaD Lab, FAU, Digital Health and Gait Analysis Group".format(datetime.now().year)

# -- Copy the README and Changelog and fix links --------------------------------------


def convert_github_links(base_url, text):
    regex = base_url + r"/(pull|issues|commit)/(\w+)"

    def substitute(matchobj):
        if matchobj.group(1) == "commit":
            return "[{}]({})".format(matchobj.group(2)[:5], matchobj.group(0))
        return "[#{}]({})".format(matchobj.group(2), matchobj.group(0))

    return re.sub(regex, substitute, text)


HERE = Path(__file__).parent
with (HERE.parent / "README.md").open() as f:
    out = f.read()
with (HERE / "README.md").open("w+") as f:
    f.write(out)

with (HERE.parent / "CHANGELOG.md").open() as f:
    out = f.read()
with (HERE / "CHANGELOG.md").open("w+") as f:
    f.write(convert_github_links(URL, out))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "recommonmark",
]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Taken from sklearn config
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/" "tex-chtml.js"

autodoc_default_options = {"members": True, "inherited-members": True, "special_members": True}
# autodoc_typehints = 'description'  # Does not work as expected. Maybe try at future date again

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Activate the theme.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/mad-lab-fau/tpcp",
    "show_prev_next": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Options for extensions --------------------------------------------------
# Intersphinx

# intersphinx configuration
intersphinx_module_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:25.0) Gecko/20100101 Firefox/25.0"

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    **intersphinx_module_mapping,
}

# Sphinx Gallary
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["./auto_examples"],
    "reference_url": {"tpcp": None, **{k: v[0] for k, v in intersphinx_module_mapping.items()}},
    # 'default_thumb_file': 'fig/logo.png',
    "backreferences_dir": "modules/generated/backreferences",
    "doc_module": ("tpcp",),
    "filename_pattern": re.escape(os.sep),
    "remove_config_comments": True,
    "show_memory": True,
    "subsection_order": ExplicitOrder(
        [
            "../examples/datasets",
            "../examples/algorithms",
            "../examples/parameter_optimization",
            "../examples/validation",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
}


linkcode_resolve = make_linkcode_resolve(
    "tpcp",
    "https://github.com/mad-lab-fau/tpcp/blob/{revision}/{package}/{path}#L{lineno}",
)


def skip_properties(app, what, name, obj, skip, options):
    """This removes all properties from the documentation as they are expected to be documented in the docstring."""
    if isinstance(obj, property):
        return True


def setup(app):
    app.connect("autodoc-skip-member", skip_properties)
