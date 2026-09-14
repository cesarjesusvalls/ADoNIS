"""build_hv_sf(probe="EM") builds the (e,e') photon hard-vertex records.

Asserted through the reweight itself rather than the record internals: the photon has no axial
current, so the EM QE weight is flat in every axial knob while the vector knobs still move it, and
the EM RES hard vertex is the identity.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.channels import ee as ee_x, res_ee as res_ee_x
from adonis.nuclear.spectral import SpectralFunction
from adonis.nuclear.targets import resolve_targets
from adonis.reweight.reweight_model import build_hv_sf, nominal_knobs
from adonis.reweight.reduced_amps2 import qe_reduced_reweight, res_reduced_reweight

_ALL = (0.0, 180.0)
_sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
_qe = ee_x.generate(256, material="C", seed=1, E_beam=1000.0, records=True, theta_acc=_ALL)
_res = res_ee_x.generate(256, material="C", seed=2, E_beam=1000.0, records=True, theta_acc=_ALL)
_HV, _SF = build_hv_sf(_qe, _res, _sf, probe="EM")


def _qe_w(**shift):
    nom = nominal_knobs()
    return np.asarray(qe_reduced_reweight(_HV["qe_reduced"], nom._replace(**shift),
                                          probe="EM", is_proton=_HV["qe_isp"]))


def test_em_records_build_without_keyerror():
    assert len(_qe["k_lep"]) > 0 and len(_res["p_N"]) > 0
    for key in ("qe_reduced", "qe_probe", "qe_isp", "res_reduced"):
        assert key in _HV, f"missing EM hard-vertex record {key}"
    assert _HV["qe_probe"] == "EM"
    for key in ("grids", "qe_pmag", "qe_erem", "res_pmag", "res_erem"):
        assert key in _SF, f"missing SF field {key}"


def test_em_is_flat_in_the_axial_knobs_and_moves_with_the_vector_ones():
    """No photon axial current -> the axial knobs cannot move the EM QE weight; the vector ones must."""
    assert np.allclose(_qe_w(), 1.0), "EM QE reweight is not 1 at nominal"
    for knob, value in (("M_A_qe", 1.15), ("axial_strength", 1.2)):
        assert np.allclose(_qe_w(**{knob: value}), 1.0), f"EM QE weight responds to axial knob {knob}"
    for knob, value in (("vector_strength", 1.1), ("gep", 1.1), ("mu_p", 1.1)):
        assert not np.allclose(_qe_w(**{knob: value}), 1.0), f"EM QE weight is flat in vector knob {knob}"


def test_em_res_hard_vertex_is_identity_by_design():
    """The EM probe carries no RES hard-vertex handle, so its RES M is zero and the guarded ratio is 1."""
    assert np.allclose(np.asarray(_HV["res_reduced"]["M"]), 0.0), "EM RES M is not zero"
    nom = nominal_knobs()
    for knob, value in (("delta_strength", 1.2), ("res_axial_strength", 1.2), ("M_A_res", 1.15)):
        w = np.asarray(res_reduced_reweight(_HV["res_reduced"], nom._replace(**{knob: value})))
        assert np.allclose(w, 1.0), f"EM RES weight responds to {knob}"
