"""Every test the per-push gate names exists, and the gate stays cheap.

The gate lists its files one by one rather than collecting a directory, so a renamed or deleted test
silently stops being checked -- and pytest exits 4 on a path it cannot find, which reads as a broken
workflow rather than as a missing test.  The heavy files carry that weight deliberately: they need
data tables or minutes of compute, and the gate installs neither.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / ".github" / "workflows" / "checks.yml"


def _listed():
    m = re.search(r"python -m pytest[^\n]*\n((?:\s+tests/[^\n]*\n)+)", GATE.read_text())
    assert m, "no pytest invocation with a file list in the gate"
    return [s.strip().rstrip(" \\") for s in m.group(1).splitlines() if s.strip()]


def test_the_gate_names_files_that_exist():
    missing = [p for p in _listed() if not (ROOT / p).exists()]
    assert not missing, f"the gate runs files that are not in the tree: {missing}"


def test_the_gate_skips_the_files_that_need_data_or_minutes():
    from tests.conftest import _HEAVY
    heavy = {f"tests/{n}" for n in _HEAVY}
    overlap = sorted(heavy.intersection(_listed()))
    assert not overlap, f"the gate installs no data tables, so it cannot run: {overlap}"
