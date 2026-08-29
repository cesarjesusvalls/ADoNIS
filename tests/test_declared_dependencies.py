"""Every third-party module adonis/ imports is declared in pyproject.toml.

An undeclared import survives in a source checkout, where the module happens to be installed for other
reasons, and fails only after `pip install adonis` on a clean machine -- at first use, not at install.
Optional extras count as declared: a module reached only through an extra is the extra's business.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(p for p in (ROOT / "adonis").rglob("*.py") if "__pycache__" not in str(p))

# import name -> distribution name, where they differ
DISTRIBUTION = {"yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn"}
FIRST_PARTY = {"adonis", "analysis"}


def _imported_roots():
    """The top-level module name of every import in adonis/, at any nesting depth."""
    roots = set()
    for path in FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _declared():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    specs = list(cfg.get("dependencies", []))
    for extra in cfg.get("optional-dependencies", {}).values():
        specs += list(extra)
    out = set()
    for spec in specs:
        name = spec.split(";")[0].split("[")[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(sep)[0]
        out.add(name.strip().lower().replace("_", "-"))
    return out


def test_every_third_party_import_is_declared():
    declared = _declared()
    missing = sorted(
        r for r in _imported_roots()
        if r not in FIRST_PARTY
        and r not in sys.stdlib_module_names
        and not r.startswith("_")
        and DISTRIBUTION.get(r, r).lower().replace("_", "-") not in declared
    )
    assert not missing, (
        "adonis/ imports these but pyproject.toml declares neither them nor an extra providing them: "
        + ", ".join(missing))
