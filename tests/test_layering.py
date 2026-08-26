"""The package may not import the application, and may not patch sys.path to do it.

adonis/ imported analysis/paper/ at 14 sites, reached by sys.path.insert in 9 files, so `import adonis`
only worked when the paper application happened to sit next to it.  Most of it was not even used --
adonis/fit/fitters.py imported nine names from analysis.paper.physical_fit and used two.

This is a test rather than a convention because a convention did not hold: adonis/fit/__init__.py
already stated the rule in prose while five modules under it broke the rule.

If this fails, the fix is never to add the module to an allow-list.  Either the thing being imported is
reusable -- move it into adonis/ -- or the importer is an application and belongs in analysis/.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "adonis"
PY_FILES = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in str(p))


def _imports(path):
    """Every module name imported by `path`, including `from X import Y` as X.Y."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        pytest.fail(f"{path} does not parse: {e}")
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_package_does_not_import_the_application(path):
    bad = sorted(m for m in _imports(path) if m == "analysis" or m.startswith("analysis."))
    assert not bad, (
        f"{path.relative_to(PKG)} imports the application layer: {bad}.\n"
        "adonis/ must stand alone.  Move the shared code into adonis/, or move the importer out."
    )


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_package_does_not_patch_sys_path(path):
    """sys.path.insert is how the inverted imports were reached; without it they fail at import time,
    which is the point -- the boundary should be enforced by the interpreter, not by a reviewer."""
    offending = [i + 1 for i, line in enumerate(path.read_text().splitlines())
                 if "sys.path.insert" in line and not line.lstrip().startswith("#")]
    assert not offending, (
        f"{path.relative_to(PKG)} patches sys.path at line(s) {offending}.  "
        "If an import needs this, the import is wrong."
    )


CAMPAIGN_ENV = ("S4_", "PHYSFIT_", "ALTGEN_", "NUTS_")


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_package_reads_no_campaign_env(path):
    src = path.read_text()
    bad = [f"{p}*" for p in CAMPAIGN_ENV if f'"{p}' in src or f"'{p}" in src]
    assert not bad, (f"{path.relative_to(PKG)} references campaign environment variables {bad}. "
                     "Per-run values belong in a config the caller passes, or in analysis/.")


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_package_names_no_study_point(path):
    """No module may hardcode a path into this paper's config or output tree."""
    src = path.read_text()
    bad = [s for s in ("configs/fits/sec", "output/altgen") if s in src]
    assert not bad, (f"{path.relative_to(PKG)} hardcodes {bad}. The package should take paths from "
                     "its caller; naming this campaign's files makes it unusable for another.")
