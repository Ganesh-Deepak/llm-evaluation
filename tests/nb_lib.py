"""Helpers shared by build_notebooks.py and nb_chapters.py."""

import textwrap

SETUP = """\
import math, os, sys, random
sys.path.insert(0, os.path.abspath(".."))      # run from notebooks/
sys.path.insert(0, os.path.abspath("."))       # or from the repo root
import numpy as np
"""

NOTEBOOKS: dict[str, list] = {}


def notebook(name):
    def deco(fn):
        NOTEBOOKS[name] = fn()
        return fn
    return deco


def md(s):
    return ("md", textwrap.dedent(s).strip("\n"))


def code(*parts):
    return ("code", "\n".join(textwrap.dedent(p).strip("\n") for p in parts))


def to_ipynb(cells):
    out = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "source": lines, "outputs": [], "execution_count": None})
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 4}


