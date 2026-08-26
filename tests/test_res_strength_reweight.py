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

from adonis.channels import res as res_xsec
from adonis.nuclear.spectral import SpectralFunction
from adonis.nuclear.targets import resolve_targets
from adonis.reweight.amps2_records import (build_res_ma_records, build_res_pw_records,
                                        build_res_pionpole_records, strength_reweight, RES_ITIZ)
from adonis.channels.dcc import current as dcc
from adonis.core.params import DCCKnobs

_tg = resolve_targets("C")[0][0]
_E = res_xsec.generate(4000, seed=1, return_events=True,
                       sf_n=SpectralFunction(_tg.spectral_n), sf_p=SpectralFunction(_tg.spectral_p),
                       n_neutron=6, n_proton=6)["events"]
_ARG = [np.asarray(_E[k]) for k in ("k_nu", "k_lep", "p_struck", "p_N", "p_pi")]
_IP = np.asarray(_E["ipid"]); _PP = np.asarray(_E["ppid"])
_REC = build_res_ma_records(*_ARG, _IP, _PP)


def test_res_axial_strength_nominal_identity():
    assert np.allclose(np.asarray(strength_reweight(_REC, 1.0)), 1.0, atol=1e-12)


def test_res_axial_strength_matches_amps2():
    """Per RES channel, w(s)*amps2(r=1) == amps2(r=s) on valid (non-placeholder) events."""
    from adonis.reweight.amps2_records import RES_ITIZ
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


# ---- pw_norm: per-partial-wave DCC norm (1+pw_norm[w]); weight = strength_reweight(rec, 1+pw) ----
_WAVES = (5, 0)                                          # P33 (Delta, dominant) + S11
_PWREC = {w: build_res_pw_records(*_ARG, _IP, _PP, w) for w in _WAVES}


def test_pw_nominal_identity():
    for w, rec in _PWREC.items():
        assert np.allclose(np.asarray(strength_reweight(rec, 1.0)), 1.0, atol=1e-12), w


def test_pw_matches_amps2():
    """For wave w, w(s)*amps2(s=1) == amps2(pw_norm[w]=s-1) per channel (s=1+pw_norm)."""
    a, b, c = (np.asarray(_PWREC[5][i]) for i in range(3))
    for w, rec in _PWREC.items():
        bb, cc = np.asarray(rec[1]), np.asarray(rec[2])
        for s in (0.6, 1.4):
            pw = [0.0] * 14; pw[w] = s - 1.0
            kn = DCCKnobs(pw_norm=tuple(pw))
            wt = np.asarray(strength_reweight(rec, s))
            for (ip, pp), itiz in RES_ITIZ.items():
                m = (_IP == ip) & (_PP == pp)
                if not m.any():
                    continue
                args = [x[m] for x in _ARG]; nn = int(m.sum())
                a1 = dcc.exclusive_amps2_batch(*args, itiz, pp)
                direct = dcc.exclusive_amps2_batch(*args, itiz, pp, knobs=kn)
                valid = np.isfinite(a1) & (a1 > 0) & ~((bb[m] == 0.0) & (cc[m] == 0.0))
                assert np.allclose((wt[m] * a1)[valid], direct[valid], rtol=1e-7), (w, s, ip, pp)


def test_pw_autodiff_equals_fd():
    eps = 1e-5
    for w, rec in _PWREC.items():
        f = lambda s: jnp.sum(strength_reweight(rec, s))
        g_ad = float(jax.grad(f)(1.1))
        g_fd = (float(f(1.1 + eps)) - float(f(1.1 - eps))) / (2 * eps)
        assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-6, w


# ---- pion_pole (induced pseudoscalar / F_P) ----
_PPREC = build_res_pionpole_records(*_ARG, _IP, _PP)


def test_pionpole_nominal_identity():
    assert np.allclose(np.asarray(strength_reweight(_PPREC, 1.0)), 1.0, atol=1e-12)


def test_pionpole_matches_amps2():
    a, b, c, _ = (np.asarray(x) for x in _PPREC)
    for s in (0.5, 1.5):
        wt = np.asarray(strength_reweight(_PPREC, s))
        for (ip, pp), itiz in RES_ITIZ.items():
            m = (_IP == ip) & (_PP == pp)
            if not m.any():
                continue
            args = [x[m] for x in _ARG]; nn = int(m.sum())
            a1 = dcc.exclusive_amps2_batch(*args, itiz, pp, pion_pole=1.0)
            direct = dcc.exclusive_amps2_batch(*args, itiz, pp, pion_pole=s)
            valid = np.isfinite(a1) & (a1 > 0) & ~((b[m] == 0.0) & (c[m] == 0.0))
            assert np.allclose((wt[m] * a1)[valid], direct[valid], rtol=1e-7), (s, ip, pp)


def test_pionpole_autodiff_equals_fd():
    f = lambda s: jnp.sum(strength_reweight(_PPREC, s))
    eps = 1e-5
    g_ad = float(jax.grad(f)(1.2))
    g_fd = (float(f(1.2 + eps)) - float(f(1.2 - eps))) / (2 * eps)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-6
