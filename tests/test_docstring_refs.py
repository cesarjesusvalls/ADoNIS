"""A module named in a docstring must exist.

Only dotted first-party paths are checked, and only where the leading segment is a real top-level
package, so ordinary prose is not mistaken for a reference.  Resolution is by finder: nothing is
imported.  tests/ is excluded, because a test's docstring may legitimately name a module it asserts
is absent.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(p for d in ("adonis", "analysis") for p in (ROOT / d).rglob("*.py")
               if "__pycache__" not in str(p))
OURS = ("adonis", "analysis")
# adonis.foo.bar -- at least two segments, so a bare "adonis" or "analysis" is not a reference
DOTTED = re.compile(r"\b((?:adonis|analysis)(?:\.[a-z_][a-z0-9_]*)+)")


def _spec(name):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None


def _binds(mod, name):
    """True if module/package `mod` binds `name` at top level."""
    spec = _spec(mod)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return spec is not None          # cannot inspect: do not manufacture a failure
    try:
        tree = ast.parse(pathlib.Path(spec.origin).read_text())
    except (OSError, SyntaxError):
        return True
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return True
        if isinstance(n, ast.Import) and any((a.asname or a.name).split(".")[0] == name for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == name for a in n.names):
            return True
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t_ in ast.walk(n):
                if isinstance(t_, ast.Name) and t_.id == name and isinstance(t_.ctx, ast.Store):
                    return True
    return "__getattr__" in pathlib.Path(spec.origin).read_text()


def _resolvable(name):
    """The full path is a module, or its parent is a module that binds the final segment.

    Walking up to ANY resolvable ancestor is not enough: `adonis.channels.res_xsec` would then pass
    on the strength of `adonis.channels` existing, which is exactly the dead reference this catches.
    """
    if _spec(name) is not None:
        return True
    parent, _, tail = name.rpartition(".")
    return bool(parent) and _binds(parent, tail)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_docstrings_name_real_modules(path):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        pytest.fail(f"{path} does not parse: {e}")
    bad = set()
    for n in ast.walk(tree):
        if not isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(n)
        if not doc:
            continue
        for ref in DOTTED.findall(doc):
            if not _resolvable(ref):
                bad.add(ref)
    assert not bad, (f"{path.relative_to(ROOT)} documents module(s) that do not exist: "
                     + ", ".join(sorted(bad)))
