"""Bit-exact JAX port of the ACHILLES Dirac spinors and gamma matrices (Spinor.cc / Spinor.hh).

WEYL (chiral) basis, Sherpa-style, with gamma5 = diag(-1,-1,1,1).  Spinors are built from the
massless WeylSpinor helicity solutions plus the massive omp/omm correction, exactly as
Spinor::Spinor(type, bar, hel, mom, ms).  Normalisation is the relativistic one: ubar.u = 2m and
sum_hel u ubar = (pslash + m) -- the tests in tests/test_xsec_spinor.py assert both bit-exact.

A spinor here is a complex128 array of shape (..., 4).  The bar contraction ubar*u is the plain
sum_i ubar[i]*u[i] (Bar() already conjugated + swapped the 2-blocks, = u^dagger gamma0 in Weyl).
All momenta are (..., 4) = (E,px,py,pz) in MeV, ACHILLES convention.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# ---- gamma matrices (row-major 4x4), exactly Spinor.hh:127-146 ------------------------------- #
_i = 1j
GAMMA = np.stack([
    np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], complex),           # g0
    np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], complex),         # g1
    np.array([[0, 0, 0, -_i], [0, 0, _i, 0], [0, _i, 0, 0], [-_i, 0, 0, 0]], complex),     # g2
    np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], complex),         # g3
])                                                                          # (4,4,4): mu,i,j
GAMMA5 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(complex)                    # Spinor.hh:144
IDENT4 = np.eye(4, dtype=complex)
PL = (IDENT4 - GAMMA5) / 2.0
PR = (IDENT4 + GAMMA5) / 2.0
METRIC = np.array([1.0, -1.0, -1.0, -1.0])

GAMMA_J = jnp.asarray(GAMMA); GAMMA5_J = jnp.asarray(GAMMA5)
PL_J = jnp.asarray(PL); PR_J = jnp.asarray(PR)

# DIAGNOSTIC flag (default False = faithful massive spinors). Set via env ADONIS_MASSLESS_LEPTON=1.
import os as _os
FORCE_MASSLESS = _os.environ.get("ADONIS_MASSLESS_LEPTON", "0") not in ("0", "", "false", "False")


def sigma_munu(mu, nu):
    """sigma^{mu nu} = (i/2)[gamma^mu, gamma^nu], the hardcoded matrices of Spinor.cc:89-112."""
    if mu == nu:
        return np.zeros((4, 4), complex)
    sign = 1
    if mu > nu:
        sign = -1; mu, nu = nu, mu
    li = 1j
    tab = {
        (0, 1): [0, -li, 0, 0, -li, 0, 0, 0, 0, 0, 0, li, 0, 0, li, 0],
        (0, 2): [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
        (0, 3): [-li, 0, 0, 0, 0, li, 0, 0, 0, 0, li, 0, 0, 0, 0, -li],
        (1, 2): [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
        (1, 3): [0, li, 0, 0, -li, 0, 0, 0, 0, 0, 0, li, 0, 0, -li, 0],
        (2, 3): [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    }
    return sign * np.array(tab[(mu, nu)], complex).reshape(4, 4)


# sigma^{mu nu} q_nu with the metric handled exactly as NuclearModel.cc:627-632
# (gamma[mu] += i F2 sigma^{mu,nu} sign q[nu]/(2 mN), sign=+1 for nu=0 else -1 = lower-index q_nu).
SIGMA = np.stack([np.stack([sigma_munu(mu, nu) for nu in range(4)]) for mu in range(4)])  # (mu,nu,4,4)
SIGMA_J = jnp.asarray(SIGMA)


def _weyl(type_, p):
    """WeylSpinor(type, p) -> (U1, U2). p is (...,4). Spinor.hh:17-24."""
    pplus = p[..., 0] + p[..., 3]
    pminus = p[..., 0] - p[..., 3]
    rpp = jnp.sqrt(pplus.astype(complex))
    rpm = jnp.sqrt(pminus.astype(complex))
    pt = p[..., 1] + 1j * p[..., 2]                          # PT
    pt_t = p[..., 1] + (1j * p[..., 2] if type_ else -1j * p[..., 2])
    # rpm = pt'/rpp where pt!=0, else sqrt(pminus)
    safe = jnp.where(jnp.abs(pt) != 0, pt_t / jnp.where(jnp.abs(rpp) > 0, rpp, 1.0), rpm)
    rpm = jnp.where(jnp.abs(pt) != 0, safe, rpm)
    return rpp, rpm


def spinor(type_, bar, hel, mom, ms=1):
    """Port of Spinor::Spinor(type,bar,hel,mom,ms) for the GENERIC (non-rest) momentum branch.
    type_: True=u, False=v.  bar: produce ubar/vbar.  hel in {-1,+1}.  mom (...,4) MeV.
    Returns the spinor components (...,4) complex."""
    mode = bool(type_) ^ (hel < 0)
    P3 = jnp.sqrt(mom[..., 1] ** 2 + mom[..., 2] ** 2 + mom[..., 3] ** 2)
    phE = jnp.where(mom[..., 0] < 0.0, -P3, P3)              # ph.E() = +/- |p3|
    ph = jnp.stack([phE, mom[..., 1], mom[..., 2], mom[..., 3]], axis=-1)
    u = [jnp.zeros(mom.shape[:-1], complex) for _ in range(4)]
    if mode:                                                 # u+(p,m)
        U1, U2 = _weyl(True, ph)
        u[2] = U1; u[3] = U2
    else:                                                    # u-(p,m)
        U1, U2 = _weyl(False, ph)
        flip = mom[..., 0] < 0.0                             # sh = -sh if E<0
        U1 = jnp.where(flip, -U1, U1); U2 = jnp.where(flip, -U2, U2)
        u[0] = U2; u[1] = -U1
    m2 = mom[..., 0] ** 2 - P3 ** 2
    sgn = 1.0 if (bool(type_) ^ (ms < 0)) else -1.0
    omp = jnp.sqrt((mom[..., 0] + phE) / (2.0 * phE))
    omm = jnp.sqrt((mom[..., 0] - phE) / (2.0 * phE))
    r = 0 if mode else 2
    # massive correction (apply where m2 != 0; massless leptons keep m2~0 -> skip)
    # DIAGNOSTIC: FORCE_MASSLESS drops the omp/omm mass correction (treats lepton as massless,
    # so q.L=0 -> contraction blind to the q^mu pion-pole longitudinal, like the old fold).
    nz = (jnp.abs(m2) > 1e-8) & (not FORCE_MASSLESS)
    new_r0 = sgn * omm * u[2 - r]; new_r1 = sgn * omm * u[3 - r]
    u_2r = u[2 - r] * omp; u_3r = u[3 - r] * omp
    out = list(u)
    out[0 + r] = jnp.where(nz, new_r0, u[0 + r])
    out[1 + r] = jnp.where(nz, new_r1, u[1 + r])
    out[2 - r] = jnp.where(nz, u_2r, u[2 - r])
    out[3 - r] = jnp.where(nz, u_3r, u[3 - r])
    s = jnp.stack(out, axis=-1)
    if bar:                                                  # Bar(): conj + swap 2-blocks
        s = jnp.stack([jnp.conj(s[..., 2]), jnp.conj(s[..., 3]),
                       jnp.conj(s[..., 0]), jnp.conj(s[..., 1])], axis=-1)
    return s


def ubar(hel, mom):  return spinor(True, True, hel, mom)
def uspinor(hel, mom):  return spinor(True, False, hel, mom)
def vbar(hel, mom):  return spinor(False, True, hel, mom)
def vspinor(hel, mom):  return spinor(False, False, hel, mom)


def sandwich(ubar_s, M, u_s):
    """ubar . M . u  (M a (...,4,4) SpinMatrix or a single (4,4)).  Returns (...,) complex."""
    Mu = jnp.einsum('...ij,...j->...i', M, u_s) if M.ndim > 2 else jnp.einsum('ij,...j->...i', M, u_s)
    return jnp.sum(ubar_s * Mu, axis=-1)
