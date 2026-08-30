"""The reference files the comparison reads satisfy the documented observable schema.

The schema is the contract between generators: ADoNIS compares against any file that satisfies it,
whichever generator wrote it.  A field silently dropped from an extractor would otherwise surface as
a wrong cross section rather than as a missing key.
"""
from __future__ import annotations

import pathlib

import pytest

from adonis.io import output_root
from adonis.oracle.schema import KINDS, problems

REFERENCE = sorted((output_root() / "achilles" / "fsrich").glob("*.npz"))


def test_every_kind_requires_weight_and_normalisation():
    for kind, fields in KINDS.items():
        assert "w" in fields and "weight_to_nb" in fields, kind


@pytest.mark.skipif(not REFERENCE, reason="no reference files present")
@pytest.mark.parametrize("path", REFERENCE, ids=lambda p: p.name)
def test_reference_file_satisfies_the_schema(path):
    issues = problems(path, "cc")
    assert not issues, f"{path.name}:\n  " + "\n  ".join(issues)


def test_a_file_missing_normalisation_is_rejected(tmp_path):
    """Written by removing the field, so the check is known to fail when the contract is broken."""
    import numpy as np
    n = 4
    good = dict(w=np.ones(n), weight_to_nb=2.0, lep=np.zeros((n, 4)),
                prot_p4=np.zeros((n, 2, 4)), pi_p4=np.zeros((n, 2, 4)), pi_pid=np.zeros((n, 2), int))
    ok = tmp_path / "ok.npz"; np.savez(ok, **good)
    assert problems(ok, "cc") == []
    bad = tmp_path / "bad.npz"; np.savez(bad, **{k: v for k, v in good.items() if k != "weight_to_nb"})
    assert any("weight_to_nb" in s for s in problems(bad, "cc"))
