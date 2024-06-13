# Configuration file for the Sphinx documentation builder.
import importlib_metadata
import sphinx_rtd_theme

project = 'ataraxis-video-system'
copyright = '2024, Ivan Kondratyev (Inkaros) & Jacob Groner & Sun Lab'
authors = ['Ivan Kondratyev (Inkaros)', 'Jacob Groner']
release = importlib_metadata.version("ataraxis-video-system")  # Extracts project version from the metadata .toml file.

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # To build documentation from docstrings
    'sphinx.ext.napoleon',  # To read google-style docstrings
    'sphinx_rtd_theme',  # To format the documentation html using ReadTheDocs format
    'sphinx_click'  # To read docstrings and command-line argument data from click-wrapped functions.
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstring parsing configuration for napoleon extension
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Directs sphinx to use RTD theme
