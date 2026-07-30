"""Regression guard for the ONE bank generator, adonis.workflow.generate_bank.generate_bank(cfg).

generate_bank replaced the three legacy drivers (reweight/event_bank + ee_event_bank -> reweight_bank,
and beams/beam_bank) with ONE GenConfig-driven backbone dispatched by adonis.workflow.cli.  Byte-for-byte
equality to every legacy generator was proven at unification time (P2/P3); this test guards the SCHEMA +
basic sanity across ALL THREE probes so a future edit can't silently drop a record type or a probe -- and
guards the ground-truth contract: banks store ONLY prim_fate + nsc_prim, with reacted/absorbed DERIVED.

Gated behind --runslow (builds real cascades at import -> JIT compile).  N is tiny.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.workflow.config import GenConfig
from adonis.workflow.generate_bank import generate_bank
from adonis.workflow import records as REC

_TMP = _os.path.join(_os.environ.get("ADONIS_SCRATCH", "/tmp"), "test_generate_bank_unify")


def _build(probe):
    if probe == "CC":
        gc = GenConfig(probe="CC", beam="spectrum", flux="t2k", material="C",
                       channels=("qe", "res"), n_per_seed=120, n_seeds=1, seed0=0, fsi=True)
    elif probe == "EM":
        gc = GenConfig(probe="EM", beam="electron", material="C", channels=("qe", "res"),
                       n_per_seed=120, n_seeds=1, seed0=0, e_beam=2222.0, fsi=True)
    else:  # hadron
        gc = GenConfig(probe="hadron", beam="pip", material="C", pmin=50.0, pmax=1000.0,
                       n_per_seed=120, n_seeds=1, seed0=0)
    out = _os.path.join(_TMP, probe)
    generate_bank(gc, out)
    return dict(np.load(f"{out}/chunk_000.npz", allow_pickle=True))


_CC = _build("CC")
_EM = _build("EM")
_HAD = _build("hadron")

# FSI kind-1 record fields (all probes) + final-state ragged fields.
_FSI = {f"f_{k}" for k in ("p_eidx", "n_eidx", "bc", "sa", "ss_el", "ss", "si", "pi_hh", "hh", "a",
                           "iso", "finel", "inel", "swap")}
_FS = {"fs_off", "fs_pid", "fs_chg", "fs_p4"}
# HARD-VERTEX amps2 records (WEAK only): the QE + RES form-factor/strength knobs.
_HV = {"hv_qe_ma_a", "hv_res_ma_a", "hv_qe_vec_a", "hv_qe_gmp_a", "hv_res_pp_a", "hv_res_delta_a"}
# THE per-primary cascade ground truth carried by EVERY probe.
_GROUND = {"prim_fate", "nsc_prim"}
_DERIVED = {"reacted", "absorbed", "captured"}   # must NOT be persisted -- derived via REC.derive_flags


def _check_ground_truth(b, wmax):
    """Every bank carries prim_fate (n, Wmax) + nsc_prim (n,), and NONE of the derived flags."""
    assert _GROUND <= set(b), sorted(_GROUND - set(b))
    assert not (_DERIVED & set(b)), f"derived flags must not be persisted: {sorted(_DERIVED & set(b))}"
    pf = b["prim_fate"]
    assert pf.ndim == 2 and pf.shape[1] == wmax, f"prim_fate shape {pf.shape} (expected (n,{wmax}))"
    fl = REC.derive_flags(b)
    n = pf.shape[0]
    assert fl["reacted"].shape == (n,) and fl["absorbed"].shape == (n,)
    assert fl["reacted"].dtype == bool and fl["absorbed"].dtype == bool
    # absorbed is a subset of reacted, by construction
    assert not (fl["absorbed"] & ~fl["reacted"]).any(), "absorbed must imply reacted"


def test_cc_schema():
    """Weak bank carries hv_* amps2 + f_* kind-1 + fs_* final state + nu kinematics; both channels; ground truth."""
    assert _FSI <= set(_CC), sorted(_FSI - set(_CC))
    assert _FS <= set(_CC), sorted(_FS - set(_CC))
    assert _HV <= set(_CC), sorted(_HV - set(_CC))
    assert {"w0", "k_nu", "p_struck", "k_lep", "prim_pi_pid", "res_p_N"} <= set(_CC)
    assert set(np.unique(_CC["channel"])) == {0, 1}, "both qe(0) and res(1) blocks expected"
    _check_ground_truth(_CC, wmax=2)                # RES has 2 primaries (pion + recoil nucleon)


def test_em_schema():
    """EM bank carries the (e,e') weight c + omega/theta + f_* + fs_*, NO hv_*; ground truth present."""
    assert _FSI <= set(_EM), sorted(_FSI - set(_EM))
    assert _FS <= set(_EM), sorted(_FS - set(_EM))
    assert {"c", "omega", "theta"} <= set(_EM)
    assert not any(k.startswith("hv_") for k in _EM), "EM bank must not carry hard-vertex amps2 records"
    assert set(np.unique(_EM["channel"])) == {0, 1}
    _check_ground_truth(_EM, wmax=2)


def test_hadron_schema():
    """Hadron beam bank: f_* + fs_* + beam_p/w0, ONE primary column, ground truth; no hard vertex."""
    assert _FSI <= set(_HAD), sorted(_FSI - set(_HAD))
    assert _FS <= set(_HAD), sorted(_FS - set(_HAD))
    assert {"beam_p", "w0"} <= set(_HAD)
    assert not any(k.startswith("hv_") for k in _HAD), "hadron bank has no hard-vertex amps2"
    _check_ground_truth(_HAD, wmax=1)                 # a beam has a single primary


def test_finite_records():
    """The stored weights + amps2 + FSI slot arrays are finite (a usable reweight bank)."""
    assert np.isfinite(_CC["w0"]).all()
    assert np.isfinite(_CC["hv_qe_ma_a"]).all() and np.isfinite(_CC["hv_res_pp_a"]).all()
    assert np.isfinite(_EM["c"]).all()
    for b in (_CC, _EM, _HAD):
        assert np.isfinite(np.asarray(b["f_sa"], np.float64)).all()
        n = len(b["prim_fate"])
        assert (b["f_p_eidx"] >= 0).all() and (b["f_p_eidx"] < n).all()
        assert (b["f_n_eidx"] >= 0).all() and (b["f_n_eidx"] < n).all()
