"""build_hv_sf(probe="EM") builds the (e,e') photon hard-vertex records.

The joint flags match what generate_bank passes for an (e,e') bank, so these tests cover the records
production actually writes rather than a combination nothing emits.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.channels import ee as ee_x, res_ee as res_ee_x
from adonis.nuclear.spectral import SpectralFunction
from adonis.nuclear.targets import resolve_targets
from adonis.reweight.reweight_model import build_hv_sf

_ALL = (0.0, 180.0)
_sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
_qe = ee_x.generate(256, material="C", seed=1, E_beam=1000.0, records=True, theta_acc=_ALL)
_res = res_ee_x.generate(256, material="C", seed=2, E_beam=1000.0, records=True, theta_acc=_ALL)


def test_em_records_build_without_keyerror():
    assert len(_qe["k_lep"]) > 0 and len(_res["p_N"]) > 0
    HV, SF = build_hv_sf(_qe, _res, _sf, with_pw=False, probe="EM", qe_joint=False, res_joint=False)
    for key in ("qe_ma", "qe_vec", "qe_gmp", "qe_gmn", "qe_gep", "qe_gen", "res_ma", "res_pp", "res_delta"):
        assert key in HV, f"missing EM hard-vertex record {key}"
    for key in ("grids", "qe_pmag", "qe_erem", "res_pmag", "res_erem"):
        assert key in SF, f"missing SF field {key}"


def test_em_axial_collapses_but_vector_carries_gradient():
    """The photon has NO axial current -> the QE M_A record is identity (b=c=0), so ma_reweight is flat.
    The QE VECTOR record must NOT be identity (the photon IS a vector current) -> real vector gradient."""
    HV, _ = build_hv_sf(_qe, _res, _sf, with_pw=False, probe="EM", qe_joint=False, res_joint=False)
    _a, b, c, _q2 = (np.asarray(x) for x in HV["qe_ma"])
    assert np.allclose(b, 0.0) and np.allclose(c, 0.0), "EM QE axial record is not identity"
    _va, vb, vc, _vq2 = (np.asarray(x) for x in HV["qe_vec"])
    assert np.any(vb != 0.0) or np.any(vc != 0.0), "EM QE vector record collapsed to identity"


def test_em_res_hard_vertex_is_identity_by_design():
    """The EM probe carries no RES hard-vertex handle, so its RES records are the identity and
    delta_strength has no EM gradient.  Asserted so a change to either shows up here."""
    HV, _ = build_hv_sf(_qe, _res, _sf, with_pw=False, probe="EM", qe_joint=False, res_joint=False)
    for key in ("res_ma", "res_pp", "res_delta"):
        a, b, c, q2 = (np.asarray(x) for x in HV[key])
        assert np.allclose(a, 1.0) and np.allclose(b, 0.0) and np.allclose(c, 0.0), f"{key} not identity"
