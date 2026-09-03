"""Every `python -m ...` the user-facing docs name resolves to a module that can be run.

The docs are the only instructions a reader has, and a module rename leaves them pointing at
something that no longer exists -- which surfaces as a failed reproduction rather than as a failed
test.  Import is not attempted: several of these pull in jax, and the point is that the name is real.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ("README.md", "analysis/paper/REPRODUCE.md")
OURS = ("adonis.", "analysis.")


def _commands():
    out = {}
    for rel in DOCS:
        text = (ROOT / rel).read_text()
        for m in re.findall(r"python -m ([A-Za-z_][A-Za-z0-9_.]*)", text):
            if m.startswith(OURS):
                out.setdefault(m, set()).add(rel)
    return out


def test_the_docs_name_at_least_the_pipeline():
    found = _commands()
    assert len(found) >= 10, f"the docs stopped naming the pipeline: {sorted(found)}"


def test_every_documented_module_exists():
    missing = []
    for name, where in sorted(_commands().items()):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            missing.append(f"{name} (named in {', '.join(sorted(where))})")
    assert not missing, "documented commands that no longer resolve:\n  " + "\n  ".join(missing)


def test_every_documented_module_is_runnable_as_a_script():
    """`python -m X` needs X to be executable, not merely importable."""
    notmain = []
    for name in sorted(_commands()):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            continue
        src = pathlib.Path(spec.origin).read_text()
        if "__main__" not in src and "def main" not in src:
            notmain.append(name)
    assert not notmain, f"named with `python -m` but has no entry point: {notmain}"
