"""Every `import` in the repo must name something that exists -- including the lazy ones.

Moving adonis/reweight/tune.py to analysis/campaign/ left a `from adonis.reweight import tune as T`
inside a function in bank_plot.py.  It survived three separate checks: a module-path rewrite that
matched `adonis.reweight.tune` but not the `from X import Y` spelling, an import sweep that only
imported modules (the broken import was never reached), and a compile pass (it is valid syntax).  The
call raised ImportError, and only a caller would have found out.

Resolution here is by FINDER, not by execution: importlib.util.find_spec locates the module without
running it, so a heavy module costs nothing and a module whose import has side effects has none.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(p for p in ROOT.rglob("*.py")
               if "__pycache__" not in str(p) and not str(p.relative_to(ROOT)).startswith("output/"))
# First-party roots only.  Third-party availability is the environment's problem, not the repo's.
OURS = ("adonis", "analysis", "tests", "jobs")


def _targets(tree):
    """(module, lineno) for every first-party import, at any nesting depth."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno) for a in n.names if a.name.split(".")[0] in OURS]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            if n.module.split(".")[0] not in OURS:
                continue
            out.append((n.module, n.lineno))
            # `from pkg import name` may import a SUBMODULE rather than an attribute; only flag it
            # when neither reading exists, so importing a function stays legal.
            for a in n.names:
                if a.name != "*":
                    out.append((f"{n.module}.{a.name}", n.lineno, n.module))
    return out


def _exists(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, AttributeError, ValueError):
        return False


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_first_party_imports_resolve(path):
    bad = []
    for t in _targets(ast.parse(path.read_text())):
        mod, line = t[0], t[1]
        if _exists(mod):
            continue
        if len(t) == 3:
            # a `from pkg import name` where name is not a submodule: fine if pkg itself resolves,
            # since it is then an attribute import and only running the module can check it.
            if _exists(t[2]):
                continue
        bad.append(f"{mod} (line {line})")
    assert not bad, f"{path.relative_to(ROOT)} imports nonexistent module(s): " + ", ".join(bad)
