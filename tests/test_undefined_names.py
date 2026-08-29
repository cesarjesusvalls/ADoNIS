"""No module may reference a global name that does not exist.

A name resolved only when a function body runs is invisible to an import check: the module loads
cleanly and fails at the call, which for an uncalled function means never.

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
FILES = sorted(p for d in ("adonis", "analysis")
               for p in (ROOT / d).rglob("*.py") if "__pycache__" not in str(p))
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__"}


def _walk_scope(node):
    """`node` and its descendants, without descending into a nested function or class body.

    ast.walk would descend, which makes every function-local name look like a module global: a name
    assigned only inside one function then read inside another would be reported as available.
    """
    stack, nested = [node], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    while stack:
        cur = stack.pop()
        yield cur
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, nested) and child is not node:
                yield child            # the def binds its own name here, but its body is its own scope
                continue
            stack.append(child)


def _bound_by(node):
    """Every name this scope binds: assignments, imports, defs, params, comprehensions, with/except/for."""
    out = set()
    for n in _walk_scope(node):
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
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        pytest.skip("star import: available names are not statically knowable")

    # `global X` inside any function binds X at module scope, and _bound_by does not descend into
    # function bodies, so those names have to be collected separately.
    declared_global = {n for node in ast.walk(tree) if isinstance(node, ast.Global) for n in node.names}
    module_scope = _bound_by(tree) | BUILTINS | declared_global
    enclosing = {}                       # function -> the names its enclosing functions bind
    def _descend(node, outer):
        for n in _walk_scope(node):
            if n is node:
                continue
            if isinstance(n, ast.ClassDef):
                # A method does not close over the class body, so the class adds nothing to `outer`;
                # but its methods still have to be reached.
                _descend(n, outer)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                enclosing[id(n)] = outer
                _descend(n, outer | _bound_by(n))
    _descend(tree, set())

    missing = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        visible = module_scope | enclosing.get(id(fn), set()) | _bound_by(fn)
        for n in _walk_scope(fn):        # a nested def is checked on its own turn, in its own scope
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in visible:
                missing.setdefault(n.id, n.lineno)
    assert not missing, (
        f"{path.relative_to(ROOT)} reads undefined name(s): "
        + ", ".join(f"{k} (line {v})" for k, v in sorted(missing.items())))
