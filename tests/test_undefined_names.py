"""No module may reference a global name that does not exist.

adonis/measurements/t2k_stv.py::load_cc1pi multiplied by NB_PER_CM2 and A_CH, neither of which was
defined in that module or anywhere else in the repo -- NB_PER_CM2 lived in adonis/workflow/data_overlay.py
and A_CH had never existed at all.  The function raised NameError on any call, and nothing noticed
because nothing had ever called it.  Import checks do not catch this: the names are resolved when the
body runs, not when the module loads.

This walks each module's AST, collects the names a function body reads, and subtracts everything that
could legitimately provide them (module globals, imports, builtins, enclosing scopes, parameters,
comprehension targets, walrus/loop/except bindings, attribute and keyword-argument positions).  What
remains can only be a typo or a name that moved.
"""
from __future__ import annotations

import ast
import builtins
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(p for p in (ROOT / "adonis").rglob("*.py") if "__pycache__" not in str(p))
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__"}


def _bound_by(node):
    """Every name a scope binds: assignments, imports, defs, params, comprehensions, with/except/for."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out |= {(a.asname or a.name) for a in n.names}
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out |= set(n.names)
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_undefined_globals(path):
    tree = ast.parse(path.read_text())
    # `import *` makes the available-name set unknowable statically; skip rather than guess.
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        pytest.skip("star import: available names are not statically knowable")

    module_scope = _bound_by(tree) | BUILTINS
    missing = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        visible = module_scope | _bound_by(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in visible:
                missing.setdefault(n.id, n.lineno)
    assert not missing, (
        f"{path.relative_to(ROOT)} reads undefined name(s): "
        + ", ".join(f"{k} (line {v})" for k, v in sorted(missing.items())))
