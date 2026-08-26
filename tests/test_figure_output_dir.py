"""Every paper figure lands under style.OUTDIR, so ADONIS_PAPER_OUT redirects all of them.

Rendering a figure set somewhere other than the published directory is how a regenerated figure is
compared against the one in the paper.  Two of the validation renderers did not allow it: render_panels
joined "output/paper" itself and render_multiobs took the directory from the spec's out_path, so
ADONIS_PAPER_OUT moved four figure groups and silently overwrote the other eight.

A path literal is checked rather than a render, because rendering needs the banks.  Docstrings are
skipped: naming the default directory in prose is not a save target.  style.py is skipped because the
default belongs there, and the match is on a directory boundary so that sibling directories such as
output/paper_banks_p4 are not swept up.
"""
from __future__ import annotations

import ast
import importlib
import os
import pathlib
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "analysis" / "paper"
FILES = sorted(p for p in PAPER.rglob("*.py")
               if "__pycache__" not in str(p) and p.name != "style.py")
FIGDIR = "output/paper"


def _docstrings(tree):
    """The string nodes that are docstrings, by identity."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = n.body[0] if n.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def test_outdir_follows_the_environment():
    """style.OUTDIR is ADONIS_PAPER_OUT when set, and the documented default when not."""
    from analysis.paper import style
    with mock.patch.dict(os.environ, {"ADONIS_PAPER_OUT": "/tmp/adonis-figs-probe"}):
        assert pathlib.Path(importlib.reload(style).OUTDIR) == pathlib.Path("/tmp/adonis-figs-probe")
    env = {k: v for k, v in os.environ.items() if k != "ADONIS_PAPER_OUT"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert pathlib.Path(importlib.reload(style).OUTDIR) == pathlib.Path("output/paper")
    importlib.reload(style)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_module_writes_to_a_hardcoded_figure_dir(path):
    tree = ast.parse(path.read_text())
    skip = _docstrings(tree)
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and id(n) not in skip
           and (n.value == FIGDIR or n.value.startswith(FIGDIR + "/"))]
    assert not bad, (f"{path.relative_to(ROOT)} hardcodes a figure directory {bad} -- "
                     "join style.OUTDIR (or call style.save) so ADONIS_PAPER_OUT redirects it")
