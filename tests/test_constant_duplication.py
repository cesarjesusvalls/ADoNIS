"""A named numeric constant is defined in one module.

Two modules binding the same UPPERCASE name to a number give one name two values depending on which is
imported.  A deliberate second value must say so in its name (HBARC_ANL, not HBARC).
"""
from __future__ import annotations

import ast
import collections
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "adonis"


def _module_constants(path):
    """{NAME: value} for top-level `NAME = <number>` assignments."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {}
    out = {}
    for n in tree.body:
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)):
            continue
        name = n.targets[0].id
        if not name.isupper() or name.startswith("_"):
            continue
        v = n.value
        if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
            out[name] = v.value
        elif isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) \
                and isinstance(v.operand, ast.Constant) and isinstance(v.operand.value, (int, float)):
            out[name] = -v.operand.value
    return out


def test_no_constant_is_defined_twice():
    where = collections.defaultdict(list)
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        for name, val in _module_constants(p).items():
            where[name].append((str(p.relative_to(PKG.parent)), val))

    dupes = {k: v for k, v in where.items() if len(v) > 1}
    if not dupes:
        return
    lines = []
    for name, places in sorted(dupes.items()):
        vals = {v for _f, v in places}
        lines.append(f"{name} = {sorted(vals)}" + ("  <-- VALUES DIFFER" if len(vals) > 1 else ""))
        lines += [f"    {f} = {v}" for f, v in places]
    raise AssertionError(
        "constants defined in more than one module (import one definition, or rename the variant "
        "so its name says how it differs):\n  " + "\n  ".join(lines))
