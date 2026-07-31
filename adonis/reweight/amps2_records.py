"""Exact quadratic-amps2 reweight records for frozen QE/RES proposals -- the ONE mechanism for every
hard-vertex knob (axial mass, axial/vector strength, Sachs form factors, DCC partial-wave norms, pion
pole), NOT just M_A.

amps2 is QUADRATIC in the axial scale r (the hadron current is linear in FA/FAP for QE and
in the axial amplitudes for RES), so 3 evals (r = 0, 1, -1) give per-event (a, b, c) with
amps2(r) = a + b r + c r^2 exactly (gated to 5e-15 in tests/test_ma_reweight.py).  The
fit/overlay-time weight is then elementwise:

    w_MA = (a + b r_i + c r_i^2) / (a + b + c),
    r_i  = F_A_dipole(Q2_i; MA) / F_A_dipole(Q2_i; 1.0)   (axial_reweight_dipole)

-- the established M_A knob; w_MA = 1 exactly at MA = 1.0 GeV.  Q2_i is the Q2 at which
each path evaluates the form factor: QE = the ORIGINAL leptonic Q2 (dirac.py); RES = the
de-Forest-shifted amplitude Q2 (exclusive_amps2_batch return_q2)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels.currents.matrix_element import me_cross_section
from adonis.channels.dcc.form_factors import axial_reweight_dipole

# RES channel constants for the amps2 re-evaluation: (ipid, ppid) -> itiz
RES_ITIZ = {(2112, 211): -1, (2112, 111): -1, (2212, 211): +1}


def build_qe_ma_records(k_nu, k_lep, p_struck, p_out, probe="CC", is_proton=None):
    """Per-event (a, b, c, Q2) for the QE sample.  Rejected draws (proposal weight 0) sit in
    the arrays with unphysical kinematics (negative Q2, NaN amps2); they get the identity
    record (w_MA = 1, grad 0).  probe="EM": the photon has no axial current, so amps2 is independent of
    axial_scale -> (b,c)=(0,0) -> ma_reweight/strength_reweight collapse to identity (no axial gradient on
    an electron bank, which is physically correct)."""
    kn, km, ps, po = (jnp.asarray(x) for x in (k_nu, k_lep, p_struck, p_out))
    _mc = lambda **kw: me_cross_section(kn, km, ps, po, probe=probe, is_proton=is_proton, **kw)["amps2"]
    a1 = np.asarray(_mc(axial_scale=1.0)); a0 = np.asarray(_mc(axial_scale=0.0)); am = np.asarray(_mc(axial_scale=-1.0))
    q = np.asarray(kn - km); q2 = np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2     # MeV^2
    ok = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(am) & (q2 > 0)
    return (np.where(ok, a0, 1.0), np.where(ok, 0.5 * (a1 - am), 0.0),
            np.where(ok, 0.5 * (a1 + am) - a0, 0.0), np.where(ok, q2, 1.0))


def build_qe_vector_records(k_nu, k_lep, p_struck, p_out, probe="CC", is_proton=None):
    """Per-event (a, b, c, Q2) for the QE VECTOR-current scale (F1,F2 multiplier): amps2 is quadratic in
    vector_scale, so 3 evals (v=0,1,-1) give (a,b,c).  Same decomposition as the axial record; the weight
    is vector_strength_reweight(rec, v) = strength_reweight (flat-scale ratio).  Q2 carried for bookkeeping.
    probe="EM" (+ per-event is_proton) builds the photon vector record for an (e,e') bank."""
    kn, km, ps, po = (jnp.asarray(x) for x in (k_nu, k_lep, p_struck, p_out))
    _mc = lambda **kw: me_cross_section(kn, km, ps, po, probe=probe, is_proton=is_proton, **kw)["amps2"]
    a1 = np.asarray(_mc(vector_scale=1.0)); a0 = np.asarray(_mc(vector_scale=0.0)); am = np.asarray(_mc(vector_scale=-1.0))
    q = np.asarray(kn - km); q2 = np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2     # MeV^2
    ok = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(am) & (q2 > 0)
    return (np.where(ok, a0, 1.0), np.where(ok, 0.5 * (a1 - am), 0.0),
            np.where(ok, 0.5 * (a1 + am) - a0, 0.0), np.where(ok, q2, 1.0))


def build_qe_ff_records(k_nu, k_lep, p_struck, p_out, key, probe="CC", is_proton=None):
    """Per-event (a, b, c, Q2) for a single Sachs-FF scale knob `key` in {gep,gen,gmp,gmn}: amps2 is
    quadratic in the scale (3-eval at s=0,1,-1).  mu_p == 'gmp', mu_n == 'gmn'.  Weight = strength_reweight.
    probe="EM" (+ per-event is_proton) builds the (e,e') photon-vertex FF record."""
    kn, km, ps, po = (jnp.asarray(x) for x in (k_nu, k_lep, p_struck, p_out))
    def ev(s):
        return np.asarray(me_cross_section(kn, km, ps, po, ff_scale={key: s}, probe=probe, is_proton=is_proton)["amps2"])
    a1, a0, am = ev(1.0), ev(0.0), ev(-1.0)
    q = np.asarray(kn - km); q2 = np.sum(q[:, 1:] ** 2, axis=1) - q[:, 0] ** 2
    ok = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(am) & (q2 > 0)
    return (np.where(ok, a0, 1.0), np.where(ok, 0.5 * (a1 - am), 0.0),
            np.where(ok, 0.5 * (a1 + am) - a0, 0.0), np.where(ok, q2, 1.0))


def build_res_ma_records(k_nu, k_lep, p_struck, p_N, p_pi, ipid, ppid):
    """Per-event (a, b, c, Q2) for the RES sample (channel = (ipid, ppid))."""
    from adonis.channels.dcc import current as dcc
    n = len(np.asarray(k_nu))
    A = np.ones(n); B = np.zeros(n); Cq = np.zeros(n); Q2r = np.ones(n)
    ipid = np.asarray(ipid); ppid = np.asarray(ppid)
    for (ip, pp), itiz in RES_ITIZ.items():
        m = (ipid == ip) & (ppid == pp)
        if not m.any():
            continue
        args = [np.asarray(x)[m] for x in (k_nu, k_lep, p_struck, p_N, p_pi)]
        nn = int(m.sum())
        r1, q2 = dcc.exclusive_amps2_batch(*args, itiz, pp, r_axial=np.ones(nn), return_q2=True)
        r0 = dcc.exclusive_amps2_batch(*args, itiz, pp, r_axial=np.zeros(nn))
        rm = dcc.exclusive_amps2_batch(*args, itiz, pp, r_axial=-np.ones(nn))
        A[m] = r0; B[m] = 0.5 * (r1 - rm); Cq[m] = 0.5 * (r1 + rm) - r0; Q2r[m] = q2
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(Cq) & (Q2r > 0)
    return (np.where(ok, A, 1.0), np.where(ok, B, 0.0), np.where(ok, Cq, 0.0), np.where(ok, Q2r, 1.0))


def build_res_pw_records(k_nu, k_lep, p_struck, p_N, p_pi, ipid, ppid, wave, npw=14):
    """Per-event (a, b, c, Q2) for a single DCC partial-wave norm `wave` in [0,npw).  amps2 is quadratic
    in s = 1 + pw_norm[wave] (the wave's amplitude is scaled by s), so 3 numeric evals at s=0,1,2 give
    (a,b,c) with amps2(s)=a+b s+c s^2.  Weight = strength_reweight(rec, 1 + pw_norm[wave]); ==1 at 0."""
    from adonis.channels.dcc import current as dcc
    from adonis.core.params import DCCKnobs
    n = len(np.asarray(k_nu))
    A = np.ones(n); B = np.zeros(n); Cq = np.zeros(n); Q2r = np.ones(n)
    ipid = np.asarray(ipid); ppid = np.asarray(ppid)

    def _pwknob(s):
        pw = [0.0] * npw; pw[wave] = s - 1.0          # pw_norm[wave] = s-1 -> scale s on that wave
        return DCCKnobs(pw_norm=tuple(pw))

    for (ip, pp), itiz in RES_ITIZ.items():
        m = (ipid == ip) & (ppid == pp)
        if not m.any():
            continue
        args = [np.asarray(x)[m] for x in (k_nu, k_lep, p_struck, p_N, p_pi)]
        e1, q2 = dcc.exclusive_amps2_batch(*args, itiz, pp, knobs=_pwknob(1.0), return_q2=True)
        e0 = dcc.exclusive_amps2_batch(*args, itiz, pp, knobs=_pwknob(0.0))
        e2 = dcc.exclusive_amps2_batch(*args, itiz, pp, knobs=_pwknob(2.0))
        # amps2(s)=a+b s+c s^2 from s=0,1,2:  a=e0, c=(e2-2e1+e0)/2, b=e1-e0-c
        cc = 0.5 * (e2 - 2 * e1 + e0); bb = e1 - e0 - cc
        A[m] = e0; B[m] = bb; Cq[m] = cc; Q2r[m] = q2
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(Cq) & (Q2r > 0)
    return (np.where(ok, A, 1.0), np.where(ok, B, 0.0), np.where(ok, Cq, 0.0), np.where(ok, Q2r, 1.0))


def build_res_pionpole_records(k_nu, k_lep, p_struck, p_N, p_pi, ipid, ppid):
    """Per-event (a, b, c, Q2) for the RES pion-pole (induced-pseudoscalar / F_P) scale.  amps2 is
    quadratic in pion_pole (it scales the longitudinal pion-pole term added to zmtx), 3-eval at {0,1,2}.
    Weight = strength_reweight(rec, pion_pole); == 1 at pion_pole = 1.0 (nominal)."""
    from adonis.channels.dcc import current as dcc
    n = len(np.asarray(k_nu))
    A = np.ones(n); B = np.zeros(n); Cq = np.zeros(n); Q2r = np.ones(n)
    ipid = np.asarray(ipid); ppid = np.asarray(ppid)
    for (ip, pp), itiz in RES_ITIZ.items():
        m = (ipid == ip) & (ppid == pp)
        if not m.any():
            continue
        args = [np.asarray(x)[m] for x in (k_nu, k_lep, p_struck, p_N, p_pi)]
        e1, q2 = dcc.exclusive_amps2_batch(*args, itiz, pp, pion_pole=1.0, return_q2=True)
        e0 = dcc.exclusive_amps2_batch(*args, itiz, pp, pion_pole=0.0)
        e2 = dcc.exclusive_amps2_batch(*args, itiz, pp, pion_pole=2.0)
        cc = 0.5 * (e2 - 2 * e1 + e0); bb = e1 - e0 - cc
        A[m] = e0; B[m] = bb; Cq[m] = cc; Q2r[m] = q2
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(Cq) & (Q2r > 0)
    return (np.where(ok, A, 1.0), np.where(ok, B, 0.0), np.where(ok, Cq, 0.0), np.where(ok, Q2r, 1.0))


def ma_reweight(rec, MA):
    """Exact per-event M_A weight from an (a, b, c, Q2) record; pure in MA, == 1 at MA = 1.0."""
    a, b, c, q2 = (jnp.asarray(x) for x in rec)
    r = axial_reweight_dipole(q2, MA)
    den = a + b + c
    return jnp.where(den > 0, (a + b * r + c * r * r) / jnp.where(den > 0, den, 1.0), 1.0)


def strength_reweight(rec, strength):
    """Exact overall-axial-STRENGTH weight from the SAME (a, b, c) amps2 record as M_A; pure in
    `strength`, == 1 at strength = 1.0.  amps2(s) = a + b*s + c*s^2 (the hadron current is linear in the
    axial block, so amps2 is exactly quadratic in the overall axial scale), so the reweight is the flat-
    scale ratio -- the M_A knob is the SAME decomposition with the Q2-dependent dipole ratio r in place of
    the flat s.  Differentiable knob: adonis/core/params.py axial_strength."""
    a, b, c, _ = (jnp.asarray(x) for x in rec)
    den = a + b + c
    return jnp.where(den > 0, (a + b * strength + c * strength * strength) / jnp.where(den > 0, den, 1.0), 1.0)
