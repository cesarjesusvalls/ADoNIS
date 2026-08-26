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
            for a in n.names:
                if a.name != "*":
                    out.append((f"{n.module}.{a.name}", n.lineno, n.module))
    return out


def _package_binds(pkg, name):
    """True if `pkg` binds `name` at module level (import, assignment, def or class).

    `pkg` may be a module or a package; spec.origin points at the .py or at the __init__.py, and
    either way a top-level binding of `name` is what makes `from pkg import name` legal.
    """
    spec = None
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, AttributeError, ValueError):
        return False
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return True
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
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t_ in ast.walk(n):
                if isinstance(t_, ast.Name) and t_.id == name and isinstance(t_.ctx, ast.Store):
                    return True
        if isinstance(n, ast.If):
            for t_ in ast.walk(n):
                if isinstance(t_, ast.Name) and t_.id == name and isinstance(t_.ctx, ast.Store):
                    return True
    return False


def _lazy_package(pkg):
    """Packages that resolve names at runtime via module __getattr__ cannot be checked statically."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, AttributeError, ValueError):
        return False
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return True
    try:
        return "__getattr__" in pathlib.Path(spec.origin).read_text()
    except OSError:
        return True


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
            pkg, name = t[2], mod.rsplit(".", 1)[1]
            if _package_binds(pkg, name) or _lazy_package(pkg):
                continue
        bad.append(f"{mod} (line {line})")
    assert not bad, f"{path.relative_to(ROOT)} imports nonexistent module(s): " + ", ".join(bad)
