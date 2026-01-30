# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Incase the project was not installed
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(".."))
master_doc = 'index'

# -- Project information -----------------------------------------------------

project = "PyEF"
copyright = '{}, <a href="https://kastner.io/en/">Melissa Manetsch and David W. Kastner</a>'.format(
    datetime.datetime.now().year
)
author = "Melissa Manetsch and David W. Kastner"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "autoapi.extension",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "revitron_sphinx_theme",
]

add_module_names = False

autosummary_generate = True
autoapi_type = "python"
autoapi_dirs = ["../pyef"]
autoapi_ignore = ["*/tests/*", "*_version.py"]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["molSimplify", "click"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]


napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme = "revitron_sphinx_theme"
html_theme_options = {
    "navigation_depth": 5,
    "github_url": "https://github.com/davidkastner/pyef",
    "color_scheme": "dark",
}

# html_logo = 'https://raw.githubusercontent.com/davidkastner/pyef/main/docs/_static/logo-white.svg'
# html_favicon = 'https://raw.githubusercontent.com/davidkastner/pyef/main/docs/_static/favicon.ico'
html_logo = '_static/logo-white.svg'
html_favicon = '_static/favicon.ico'
templates_path = ['_templates']


html_context = {
    "landing_page": {
        "menu": [{
            "title": "pyef",
             "url": "https://pyef.readthedocs.io/"
             }]
    }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ['custom.css']
html_js_files = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.

html_sidebars = {}
