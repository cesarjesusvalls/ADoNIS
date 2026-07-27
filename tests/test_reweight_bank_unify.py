"""Regression guard for the UNIFIED reweight-bank generator (adonis.workflow.reweight_bank).

generate_reweight_bank replaced the two retired env-driven generators (reweight/event_bank.py [weak] +
analysis/beams/ee_event_bank.py [EM]) with ONE GenConfig-driven driver dispatched by adonis.workflow.cli
--reweight-bank.  The byte-for-byte equality to both legacy generators was proven at unification time;
this test guards the SCHEMA + basic sanity so a future edit can't silently drop a record type or a probe.

Gated behind --runslow (builds a real QE+RES cascade at import -> JIT compile).  N is tiny.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.workflow.config import GenConfig
from adonis.workflow.reweight_bank import generate_reweight_bank

_TMP = _os.path.join(_os.environ.get("ADONIS_SCRATCH", "/tmp"), "test_reweight_bank_unify")


def _build(probe):
    if probe == "weak":
        gc = GenConfig(probe="weak", beam="spectrum", flux="t2k", material="C",
                       channels=("qe", "res"), n_per_seed=120, n_seeds=1, seed0=0, fsi=True)
    else:
        gc = GenConfig(probe="EM", beam="electron", material="C", channels=("qe", "res"),
                       n_per_seed=120, n_seeds=1, seed0=0, e_beam=2222.0, fsi=True)
    out = _os.path.join(_TMP, probe)
    generate_reweight_bank(gc, out)
    return dict(np.load(f"{out}/chunk_000.npz", allow_pickle=True))


_WEAK = _build("weak")
_EM = _build("EM")

# FSI kind-1 record fields (both probes) + final-state ragged fields (both probes).
_FSI = {f"f_{k}" for k in ("p_eidx", "n_eidx", "bc", "sa", "ss_el", "ss", "si", "pi_hh", "hh", "a",
                           "iso", "finel", "inel", "swap")}
_FS = {"fs_off", "fs_pid", "fs_chg", "fs_p4"}
# HARD-VERTEX amps2 records (WEAK only): the QE + RES form-factor/strength knobs.
_HV = {"hv_qe_ma_a", "hv_res_ma_a", "hv_qe_vec_a", "hv_qe_gmp_a", "hv_res_pp_a", "hv_res_delta_a"}


def test_weak_schema():
    """Weak bank carries hv_* amps2 + f_* kind-1 + fs_* final state + nu kinematics; both channels present."""
    assert _FSI <= set(_WEAK), sorted(_FSI - set(_WEAK))
    assert _FS <= set(_WEAK), sorted(_FS - set(_WEAK))
    assert _HV <= set(_WEAK), sorted(_HV - set(_WEAK))
    assert {"w0", "k_nu", "p_struck", "k_mu", "prim_pi_pid", "res_p_N"} <= set(_WEAK)
    ch = _WEAK["channel"]
    assert set(np.unique(ch)) == {0, 1}, "both qe(0) and res(1) blocks expected"


def test_em_schema():
    """EM bank carries the (e,e') weight c + omega/theta + f_* + fs_*, and NO hv_* (EM has no amps2 record)."""
    assert _FSI <= set(_EM), sorted(_FSI - set(_EM))
    assert _FS <= set(_EM), sorted(_FS - set(_EM))
    assert {"c", "omega", "theta"} <= set(_EM)
    assert not any(k.startswith("hv_") for k in _EM), "EM bank must not carry hard-vertex amps2 records"
    assert set(np.unique(_EM["channel"])) == {0, 1}


def test_finite_records():
    """The stored weights + amps2 + FSI slot arrays are finite (a usable reweight bank)."""
    assert np.isfinite(_WEAK["w0"]).all()
    assert np.isfinite(_WEAK["hv_qe_ma_a"]).all() and np.isfinite(_WEAK["hv_res_pp_a"]).all()
    assert np.isfinite(_EM["c"]).all()
    for b in (_WEAK, _EM):
        assert np.isfinite(np.asarray(b["f_sa"], np.float64)).all()
        # per-slot event indices point inside the event range
        n = len(b["channel"])
        assert (b["f_p_eidx"] >= 0).all() and (b["f_p_eidx"] < n).all()
        assert (b["f_n_eidx"] >= 0).all() and (b["f_n_eidx"] < n).all()
