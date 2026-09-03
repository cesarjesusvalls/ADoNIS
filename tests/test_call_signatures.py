"""A keyword argument at a call site must exist in the callee's signature.

Resolution is static: the callee module is located from the importing file's own imports and parsed,
never executed.  Only first-party `alias.func(...)` calls and same-file functions are checked, since
those are the ones whose definition can be found with certainty; a callee taking **kwargs is skipped.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

from tests._sources import ROOT, python_files

FILES = python_files()
OURS = ("adonis", "analysis")
_CACHE: dict[str, dict] = {}


def _module_funcs(mod):
    """{name: (params, has_kwargs)} for every top-level def in a first-party module."""
    if mod in _CACHE:
        return _CACHE[mod]
    out = {}
    try:
        spec = importlib.util.find_spec(mod)
        origin = spec.origin if spec else None
    except (ImportError, AttributeError, ValueError):
        origin = None
    if origin and origin.endswith(".py"):
        try:
            tree = ast.parse(pathlib.Path(origin).read_text())
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = n.args
                    names = {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs}
                    out[n.name] = (names, a.kwarg is not None)
        except (OSError, SyntaxError):
            pass
    _CACHE[mod] = out
    return out


def _alias_map(tree):
    """alias -> first-party module, from `import a.b as c` and `from a import b as c`."""
    m = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in OURS and a.asname:
                    m[a.asname] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            if n.module.split(".")[0] not in OURS:
                continue
            for a in n.names:
                if a.name != "*":
                    m[a.asname or a.name] = f"{n.module}.{a.name}"
    return m


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_call_keywords_exist(path):
    tree = ast.parse(path.read_text())
    aliases = _alias_map(tree)
    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        base = n.func.value
        if not isinstance(base, ast.Name) or base.id not in aliases:
            continue
        funcs = _module_funcs(aliases[base.id])
        sig = funcs.get(n.func.attr)
        if sig is None:
            continue
        params, has_kwargs = sig
        if has_kwargs:
            continue
        for kw in n.keywords:
            if kw.arg and kw.arg not in params:
                bad.append(f"{base.id}.{n.func.attr}({kw.arg}=...) at line {n.lineno}")
    assert not bad, (f"{path.relative_to(ROOT)} passes keyword(s) the callee does not accept: "
                     + "; ".join(bad))
