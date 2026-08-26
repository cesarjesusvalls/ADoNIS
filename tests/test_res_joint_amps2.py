"""Gates for the flexible reduced-quadratic RES hard-vertex reweight (adonis.reweight.reduced_amps2).

RES amps2 is an exact quadratic form g^T M g over the atoms {V,A,P}x{rest,wave5}; the paper's RES dials
(M_A_res, res_axial_strength, pion_pole, delta_strength) map to the atom coefficients g(knobs) with the nested
r_axial (axial+pole) / pion_pole (pole) / delta (wave5) structure.  Gates: exactness vs direct amps2 for
arbitrary simultaneous variation; nominal identity; mixed 2nd derivative nonzero & correct.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.channels import res as res_xsec
from adonis.nuclear.spectral import SpectralFunction
from adonis.nuclear.targets import resolve_targets
from adonis.channels.dcc import current as dcc
from adonis.channels.dcc.form_factors import axial_reweight_dipole
from adonis.core.params import DCCKnobs, nominal_knobs
from adonis.reweight.reduced_amps2 import build_res_reduced, res_g, res_reduced_reweight, _RES_ITIZ

_tg = resolve_targets("C")[0][0]
_E = res_xsec.generate(4000, seed=1, return_events=True,
                       sf_n=SpectralFunction(_tg.spectral_n), sf_p=SpectralFunction(_tg.spectral_p),
                       n_neutron=6, n_proton=6)["events"]
_ARG = [np.asarray(_E[k]) for k in ("k_nu", "k_lep", "p_struck", "p_N", "p_pi")]
_IP, _PP = np.asarray(_E["ipid"]), np.asarray(_E["ppid"])
_REC = build_res_reduced(*_ARG, _IP, _PP)
_M, _Q2 = np.asarray(_REC["M"]), np.asarray(_REC["Q2"])


def _direct(args_m, itiz, pp, Q2m, mar, ras, ppl, dl):
    """Direct exclusive amps2 at these RES knobs: r_axial=dipole(Q2;M_A_res)*res_axial; pion_pole; wave5*delta."""
    dip = np.asarray(axial_reweight_dipole(jnp.asarray(Q2m), mar))
    pw = [0.0] * 14; pw[dcc._DELTA_WAVE] = dl - 1.0
    return np.asarray(dcc.exclusive_amps2_batch(*args_m, itiz, pp, r_axial=dip * ras, pion_pole=ppl,
                                                knobs=DCCKnobs(pw_norm=tuple(pw))))


_TRIALS = [(1.0, 1.0, 1.0, 1.0), (1.05, 0.9, 1.1, 1.0), (1.0, 1.2, 0.8, 1.3), (1.05, 0.85, 1.15, 0.7)]


def test_res_mij_exact():
    """g^T M g reproduces the true RES amps2 for arbitrary (M_A_res, res_axial, pion_pole, delta), per channel."""
    nom = nominal_knobs()
    for (ip, pp), itiz in _RES_ITIZ.items():
        m = (_IP == ip) & (_PP == pp)
        if not m.any():
            continue
        args_m = [x[m] for x in _ARG]; Q2m = _Q2[m]; Mm = jnp.asarray(_M[m])
        a_nom = _direct(args_m, itiz, pp, Q2m, 1.0, 1.0, 1.0, 1.0)
        valid = np.isfinite(a_nom) & (a_nom > 0)
        for (mar, ras, ppl, dl) in _TRIALS:
            k = nom._replace(M_A_res=mar, res_axial_strength=ras, pion_pole=ppl, delta_strength=dl)
            lhs = np.asarray(jnp.einsum('ni,nij,nj->n', res_g(jnp.asarray(Q2m), k), Mm, res_g(jnp.asarray(Q2m), k)))
            rhs = _direct(args_m, itiz, pp, Q2m, mar, ras, ppl, dl)
            assert np.allclose(lhs[valid], rhs[valid], rtol=1e-5), (ip, pp, mar, ras, ppl, dl)


def test_res_reduced_nominal_identity():
    assert np.allclose(np.asarray(res_reduced_reweight(_REC, nominal_knobs())), 1.0, atol=1e-6)


def test_res_reduced_mixed_second_derivative():
    """d2 log(sum w)/d(res_axial_strength) d(pion_pole) at nominal: nonzero (pole rides on axial) and finite."""
    nom = nominal_knobs()
    good = jnp.asarray(np.isfinite(_Q2) & (np.abs(_M).sum(axis=(1, 2)) > 0))

    def logsumw(ras, ppl):
        w = res_reduced_reweight(_REC, nom._replace(res_axial_strength=ras, pion_pole=ppl))
        return jnp.log(jnp.sum(jnp.where(good, w, 0.0)))
    d2 = float(jax.grad(jax.grad(logsumw, argnums=0), argnums=1)(1.0, 1.0))
    assert np.isfinite(d2) and abs(d2) > 1e-8, d2


from adonis.reweight.amps2_records import (build_res_ma_records, build_res_pionpole_records,
                                        build_res_pw_records, strength_reweight, ma_reweight)

_MAREC = build_res_ma_records(*_ARG, _IP, _PP)
_PPREC = build_res_pionpole_records(*_ARG, _IP, _PP)
_DLREC = build_res_pw_records(*_ARG, _IP, _PP, dcc._DELTA_WAVE)
_GOOD = np.isfinite(_Q2) & (np.abs(_M).sum(axis=(1, 2)) > 0)


def _old_hv_res(k):
    return (ma_reweight(_MAREC, k.M_A_res) * strength_reweight(_MAREC, k.res_axial_strength)
            * strength_reweight(_PPREC, k.pion_pole) * strength_reweight(_DLREC, k.delta_strength))


_RES_SINGLE = [("M_A_res", 1.05), ("res_axial_strength", 0.85), ("pion_pole", 1.2), ("delta_strength", 0.9)]


def test_res_single_knob_matches_legacy():
    nom = nominal_knobs()
    for fld, val in _RES_SINGLE:
        wn = np.asarray(res_reduced_reweight(_REC, nom._replace(**{fld: val})))
        wo = np.asarray(_old_hv_res(nom._replace(**{fld: val})))
        assert np.allclose(wn[_GOOD], wo[_GOOD], rtol=1e-4), fld


def test_res_first_order_invariance():
    nom = nominal_knobs(); keep = jnp.asarray(_GOOD)
    for fld, _ in _RES_SINGLE:
        x0 = float(getattr(nom, fld))
        fn = lambda x: jnp.sum(jnp.where(keep, res_reduced_reweight(_REC, nom._replace(**{fld: x})), 0.0))
        fo = lambda x: jnp.sum(jnp.where(keep, _old_hv_res(nom._replace(**{fld: x})), 0.0))
        gn, go = float(jax.grad(fn)(x0)), float(jax.grad(fo)(x0))
        assert abs(gn - go) / max(abs(go), 1e-30) < 1e-4, (fld, gn, go)


def test_hv_res_dispatch():
    from adonis.reweight.reweight_model import _hv_res
    HV = {"res_reduced": _REC}
    k = nominal_knobs()._replace(res_axial_strength=1.1, pion_pole=0.9)
    assert np.allclose(np.asarray(_hv_res(k, HV)), np.asarray(res_reduced_reweight(_REC, k)), rtol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
