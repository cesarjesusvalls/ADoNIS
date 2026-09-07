"""A bank must carry the reduced-quadratic hard-vertex records, not per-knob ones.

The per-knob (a,b,c) form reweights each knob independently, so it cannot represent interference
between correlated knobs: the mixed second derivatives are identically zero.  For the RES block --
M_A_res, res_axial_strength, delta_strength -- that understates the degeneracy and hence the fitted
error.  The reduced quadratic g^T M g keeps every cross term, so it is the only supported form.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_build_hv_sf_has_no_legacy_switch():
    """A switch is a way to generate a bank that cannot be reweighted correctly."""
    from adonis.reweight.reweight_model import build_hv_sf
    params = set(inspect.signature(build_hv_sf).parameters)
    assert not (params & {"qe_joint", "res_joint", "with_pw"}), f"legacy switch survives: {params}"


def test_generator_stores_the_quadratic_records():
    src = (ROOT / "adonis" / "workflow" / "generate_bank.py").read_text()
    assert "hv_qe_mij" in src and "hv_res_mij" in src, "the generator does not store the M records"
    assert "hv_qe_ma_a" not in src, "the generator still writes per-knob records"


def test_reweighting_a_legacy_bank_raises():
    """Silently reweighting a per-knob bank is the failure this guards against."""
    from adonis.reweight.bank_reweight import bank_weight
    from adonis.reweight.reweight_model import nominal_knobs
    n = 4
    legacy = {"w0": np.ones(n), "channel": np.zeros(n, np.int8),
              "p_struck": np.zeros((n, 4), np.float32),
              "hv_qe_ma_a": np.ones(n), "hv_qe_ma_b": np.zeros(n),
              "hv_qe_ma_c": np.zeros(n), "hv_qe_ma_Q2": np.ones(n)}
    with pytest.raises(KeyError, match="reduced-quadratic"):
        bank_weight(legacy, nominal_knobs(), None)
