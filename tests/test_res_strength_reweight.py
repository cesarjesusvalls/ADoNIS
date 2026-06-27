"""Closure for the RES axial-strength knob (C5A / overall RES axial scale), reusing the RES M_A amps2
records (build_res_ma_records, the r_axial 3-eval decomposition):
1. nominal identity: strength_reweight(rec, 1.0) == 1;
2. amps2-match: w(s) * amps2(r_axial=1) == amps2(r_axial=s) per channel (the reweight IS the ratio);
3. autodiff == finite-difference.
RES axial is linear in r_axial (M_A uses the Q2-dipole ratio; C5A uses the flat scale)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.xsec import res_xsec
from adonis.xsec.spectral import SpectralFunction
from adonis.workflow.materials import resolve_targets
from adonis.analysis.ma_records import build_res_ma_records, strength_reweight
from adonis.xsec import dcc_current as dcc

_tg = resolve_targets("C")[0][0]
_E = res_xsec.generate(4000, seed=1, return_events=True,
                       sf_n=SpectralFunction(_tg.spectral_n), sf_p=SpectralFunction(_tg.spectral_p),
                       n_neutron=6, n_proton=6)["events"]
_ARG = [np.asarray(_E[k]) for k in ("k_nu", "k_mu", "p_struck", "p_N", "p_pi")]
_IP = np.asarray(_E["ipid"]); _PP = np.asarray(_E["ppid"])
_REC = build_res_ma_records(*_ARG, _IP, _PP)


def test_res_axial_strength_nominal_identity():
    assert np.allclose(np.asarray(strength_reweight(_REC, 1.0)), 1.0, atol=1e-12)


def test_res_axial_strength_matches_amps2():
    """Per RES channel, w(s)*amps2(r=1) == amps2(r=s) on valid (non-placeholder) events."""
    from adonis.analysis.ma_records import RES_ITIZ
    a, b, c, _ = (np.asarray(x) for x in _REC)
    w_full = {s: np.asarray(strength_reweight(_REC, s)) for s in (0.7, 1.3)}
    for (ip, pp), itiz in RES_ITIZ.items():
        m = (_IP == ip) & (_PP == pp)
        if not m.any():
            continue
        args = [x[m] for x in _ARG]; nn = int(m.sum())
        a1 = dcc.exclusive_amps2_batch(*args, itiz, pp, r_axial=np.ones(nn))
        valid = np.isfinite(a1) & (a1 > 0) & ~((b[m] == 0.0) & (c[m] == 0.0))
        for s in (0.7, 1.3):
            direct = dcc.exclusive_amps2_batch(*args, itiz, pp, r_axial=np.full(nn, s))
            assert np.allclose((w_full[s][m] * a1)[valid], direct[valid], rtol=1e-8), (ip, pp, s)


def test_res_axial_strength_autodiff_equals_fd():
    f = lambda s: jnp.sum(strength_reweight(_REC, s))
    eps = 1e-5
    g_ad = float(jax.grad(f)(1.2))
    g_fd = (float(f(1.2 + eps)) - float(f(1.2 - eps))) / (2 * eps)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-6
