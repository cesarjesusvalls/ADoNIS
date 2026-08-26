"""No two first-party modules import each other at module level.

An import cycle raises ImportError from whichever module the interpreter reaches second, so it breaks
every path through both modules and takes the test suite down at collection rather than at a test.
`test_imports_resolve.py` cannot see it: each import in a cycle names a module that exists, and
`find_spec` answers without executing anything.

The graph is built from top-level `import`/`from` statements only.  An import inside a function body
is deferred to call time and does not participate, which is the standard way to break a cycle
deliberately.  Edges to a parent package are not modelled, so a cycle that runs only through an
`__init__` that imports its own submodules is not reported.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OURS = ("adonis", "analysis")
FILES = sorted(p for d in OURS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in str(p))


def _module_name(path):
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts[:-1]) if rel.name == "__init__" else list(rel.parts)
    return ".".join(parts)


def _is_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _targets(tree):
    """First-party modules imported at module level, resolved to the module that actually runs."""
    out = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.split(".")[0] in OURS}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            if n.module.split(".")[0] not in OURS:
                continue
            out.add(n.module)
            # `from pkg import mod` runs pkg.mod; `from mod import name` does not.
            out |= {f"{n.module}.{a.name}" for a in n.names
                    if a.name != "*" and _is_module(f"{n.module}.{a.name}")}
    return out


def _cycles(graph):
    """One concrete cycle path per mutually-importing group of modules."""
    found, stack, on_stack, done = [], [], set(), set()

    def walk(node):
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(graph.get(node, ())):
            if nxt in on_stack:
                found.append(stack[stack.index(nxt):] + [nxt])
            elif nxt not in done:
                walk(nxt)
        stack.pop()
        on_stack.discard(node)
        done.add(node)

    for start in sorted(graph):
        if start not in done:
            walk(start)
    return found


def test_no_module_level_import_cycles():
    graph, known = {}, set()
    for p in FILES:
        mod = _module_name(p)
        known.add(mod)
        try:
            graph[mod] = _targets(ast.parse(p.read_text()))
        except SyntaxError as e:
            raise AssertionError(f"{p.relative_to(ROOT)} does not parse: {e}") from None
    graph = {m: {t for t in ts if t in known and t != m} for m, ts in graph.items()}

    cycles = _cycles(graph)
    assert not cycles, ("module-level import cycle(s) -- move the shared definition into the module "
                        "lower in the graph, or defer one import into the function that uses it:\n  "
                        + "\n  ".join(" -> ".join(c) for c in cycles))
