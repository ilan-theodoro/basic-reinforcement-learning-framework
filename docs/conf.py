"""Sphinx configuration."""
project = "Reinforcement Learning course's final project."
author = "Ilan Theodoro"
copyright = "2023, Ilan Theodoro"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
