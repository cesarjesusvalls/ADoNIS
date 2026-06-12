"""ANL-Osaka meson-baryon partial-wave cross sections (Phase E1).

Parses the ANL tables (`data/MesonBaryonAmplitudes/ANL/ANL_i-f.dat`) and forms the
physical piN -> piN total cross section by the partial-wave sum used in ACHILLES
(`MesonBaryonAmplitudes.cc::CalcCrossSectionW_grid`):

    sigma_if(W) = (hbar c)^2 * 10 * 2 pi * 4 W^2 / PF
                  * sum_{L,J} (2J+1) | sum_I CG_I^{if} A^{I}_{L,J}(W) |^2     [mb]

with PF = (W^2 - m_M^2 - m_B^2)^2 - 4 m_M^2 m_B^2 (Kallen; p_cm = sqrt(PF)/(2W)).
The table columns are the 20 waves L_{2I,2J} x (Re, Im); the label gives (L, I, J)
directly (e.g. P33 = L=1, I=3/2, J=3/2 = the Delta(1232)).

This forward sigma(W) IS the ANL-Osaka model, so it self-validates against the known
piN cross section (the Delta peak at W~1232) -- no cascade binary needed (which is a
confirmed showstopper here; see docs/phases/README.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from adonis.paths import achilles_data_root
from adonis.primary.dcc.form_factors import M_PI_GEV  # GeV; we work in MeV here

from adonis.constants import mpip as M_PI, mp as M_N  # charged pion / proton [MeV]
HBARC = 197.32              # MeV fm (verbatim local rounding of the ANL transcription)

# wave order in the ANL files; label L_{2I,2J} -> (L, twoI, twoJ)
WAVES = ["S11", "S31", "P11", "P13", "P31", "P33", "D13", "D15", "D33", "D35",
         "F15", "F17", "F35", "F37", "G17", "G19", "G37", "G39", "H19", "H39"]
_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5}


def wave_qn(name):
    """(L, twoI, twoJ) from a wave label like 'P33'."""
    return _L[name[0]], int(name[1]), int(name[2])


def load_anl(i=0, f=0, root=None):
    """Parse ANL_i-f.dat -> (W[MeV], amps[nW, 20] complex) in WAVES order."""
    root = achilles_data_root() if root is None else Path(root)
    path = root / "MesonBaryonAmplitudes" / "ANL" / f"ANL_{i}-{f}.dat"
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()])
    arr = np.array(rows)
    W = arr[:, 0]
    vals = arr[:, 1:]                                    # (nW, 40) = 20 x (Re, Im)
    amps = vals[:, 0::2] + 1j * vals[:, 1::2]            # (nW, 20)
    return W, amps


def _pcm2(W, mM=M_PI, mB=M_N):
    PF = (W ** 2 - mM ** 2 - mB ** 2) ** 2 - 4.0 * mM ** 2 * mB ** 2
    return PF                                            # ACHILLES "PF" (= 4 W^2 p_cm^2)


def _channel_sigma(amps, Wt, cg, norm=1.0):
    """Partial-wave cross section [mb] for a physical piN channel with isospin weights
    cg = {1: c_{1/2}, 3: c_{3/2}} (the products of initial+final meson-baryon Clebsches):
    sigma = pref * sum_{L,J} (2J+1) |sum_I cg_I A^I_{L,J}|^2.  Waves are paired by (L,J)."""
    # group wave indices by (L, twoJ)
    by_lj = {}
    for k, name in enumerate(WAVES):
        L, twoI, twoJ = wave_qn(name)
        by_lj.setdefault((L, twoJ), {})[twoI] = k
    s = np.zeros(len(Wt))
    for (L, twoJ), waves in by_lj.items():
        amp = np.zeros(len(Wt), dtype=complex)
        for twoI, k in waves.items():
            amp += cg.get(twoI, 0.0) * amps[:, k]
        s += (twoJ + 1.0) * np.abs(amp) ** 2
    PF = _pcm2(Wt)
    pref = HBARC ** 2 * 10.0 * 2.0 * np.pi * 4.0 * Wt ** 2 / PF
    return norm * pref * s


def pip_p_total(W=None, norm=1.0):
    """pi+ p -> pi+ p total cross section [mb] (pure I=3/2). Returns (W[MeV], sigma[mb])."""
    Wt, amps = load_anl(0, 0)
    if W is not None:
        amps = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])], axis=1)
        Wt = np.asarray(W)
    return Wt, _channel_sigma(amps, Wt, {3: 1.0}, norm)


def dsigma_dOmega(W, cos_theta, cg={3: 1.0}):
    """piN -> piN differential cross section dsigma/dOmega(theta) [arb. shape] at fixed W,
    via the standard spin-non-flip f and spin-flip g partial-wave amplitudes:

        f(theta) = sum_L [ (L+1) a_{L+} + L a_{L-} ] P_L(cos)
        g(theta) = sum_L [ a_{L+} - a_{L-} ] P_L^1(cos)        (a_{L+/-}: J = L +/- 1/2)
        dsigma/dOmega = |f|^2 + |g|^2

    a_{L,J} is the isospin-combined amplitude sum_I cg_I A^I_{L,J}.  At the Delta (W~1232)
    P33 (L=1, J=3/2=L+1/2) dominates -> the classic 1 + 3 cos^2(theta) shape.
    """
    from numpy.polynomial.legendre import Legendre
    Wt, amps = load_anl(0, 0)
    a = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])])  # (20,) at this W
    # a_{L,+/-}: index by (L, sign) where + is J=L+1/2 (twoJ=2L+1), - is J=L-1/2 (twoJ=2L-1)
    aLp, aLm = {}, {}
    for k, name in enumerate(WAVES):
        L, twoI, twoJ = wave_qn(name)
        amp = cg.get(twoI, 0.0) * a[k]
        if twoJ == 2 * L + 1:
            aLp[L] = aLp.get(L, 0j) + amp
        elif twoJ == 2 * L - 1:
            aLm[L] = aLm.get(L, 0j) + amp
    c = np.asarray(cos_theta, dtype=float)
    Lmax = 5
    PL = np.stack([np.polynomial.legendre.legval(c, [0] * L + [1]) for L in range(Lmax + 1)])
    # associated Legendre P_L^1(cos) = -sqrt(1-c^2) dP_L/dc
    s = np.sqrt(np.clip(1 - c ** 2, 0, 1))
    PL1 = np.stack([-s * _dPL(L, c) for L in range(Lmax + 1)])
    f = np.zeros_like(c, dtype=complex)
    g = np.zeros_like(c, dtype=complex)
    for L in range(Lmax + 1):
        f += ((L + 1) * aLp.get(L, 0j) + L * aLm.get(L, 0j)) * PL[L]
        g += (aLp.get(L, 0j) - aLm.get(L, 0j)) * PL1[L]
    return np.abs(f) ** 2 + np.abs(g) ** 2


def _dPL(L, c):
    """dP_L/dcos via the Legendre derivative."""
    coef = [0] * L + [1]
    d = np.polynomial.legendre.legder(coef)
    return np.polynomial.legendre.legval(c, d)


def pim_p_elastic(W=None, norm=1.0):
    """pi- p -> pi- p elastic cross section [mb].  |pi- p> = sqrt(1/3)|3/2> - sqrt(2/3)|1/2>,
    so the elastic isospin weights (initial x final) are c_{3/2}=1/3, c_{1/2}=2/3.  Smaller
    Delta peak than pi+ p (only 1/3 of the I=3/2 strength)."""
    Wt, amps = load_anl(0, 0)
    if W is not None:
        amps = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])], axis=1)
        Wt = np.asarray(W)
    return Wt, _channel_sigma(amps, Wt, {3: 1.0 / 3.0, 1: 2.0 / 3.0}, norm)


# charge-exchange isospin weights (products of initial pi-p and final pi0n Clebsches):
#   |pi- p>  = sqrt(1/3)|3/2,-1/2> - sqrt(2/3)|1/2,-1/2>
#   |pi0 n>  = sqrt(2/3)|3/2,-1/2> + sqrt(1/3)|1/2,-1/2>
#   c_{3/2} = sqrt(2/3)*sqrt(1/3) = +sqrt(2)/3 ;  c_{1/2} = sqrt(1/3)*(-sqrt(2/3)) = -sqrt(2)/3
_CEX_CG = {3: np.sqrt(2.0) / 3.0, 1: -np.sqrt(2.0) / 3.0}


def pim_p_cex(W=None, norm=1.0):
    """pi- p -> pi0 n charge-exchange cross section [mb].  At the Delta (pure I=3/2) the
    I=1/2 piece vanishes and sigma ∝ |sqrt(2)/3|^2 = 2/9 of the I=3/2 strength -> the
    textbook 9:2:1 ratio of pi+p : (pi-p->pi0n) : pi-p elastic at the Delta(1232).  Off the
    resonance the opposite-sign I=1/2 amplitude interferes, filling in the wings."""
    Wt, amps = load_anl(0, 0)
    if W is not None:
        amps = np.stack([np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])], axis=1)
        Wt = np.asarray(W)
    return Wt, _channel_sigma(amps, Wt, _CEX_CG, norm)


def pim_p_total(W=None, norm=1.0):
    """pi- p total over the two open piN final states (elastic + charge exchange) [mb]."""
    Wt, se = pim_p_elastic(W, norm)
    _, sc = pim_p_cex(W, norm)
    return Wt, se + sc


# --- inelastic production channels: piN -> etaN, KLambda (ANL_0-{1,2}) --------- #
# eta and Lambda are isoscalar, so only the I=1/2 piN amplitude contributes; the physical
# pi- p -> eta n / K0 Lambda cross section carries the isospin Clebsch (|<f|1/2><1/2|pi-p>|^2).
import numpy as _np
_R23 = _np.sqrt(2.0 / 3.0)


def eta_production(W=None, norm=1.0):
    """pi- p -> eta n cross section [mb] (ANL_0-1, I=1/2 only).  Peaks at the N(1535) (W~1535)
    which sits at the eta N threshold and couples strongly to eta N."""
    Wt, amps = load_anl(0, 1)
    if W is not None:
        amps = _np.stack([_np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])], axis=1)
        Wt = _np.asarray(W)
    return Wt, _channel_sigma(amps, Wt, {1: _R23}, norm)


def klambda_production(W=None, norm=1.0):
    """pi- p -> K0 Lambda cross section [mb] (ANL_0-2, I=1/2 only).  Threshold ~ m_K + m_Lambda
    (W~1610); the strangeness-production channel."""
    Wt, amps = load_anl(0, 2)
    if W is not None:
        amps = _np.stack([_np.interp(W, Wt, amps[:, k]) for k in range(amps.shape[1])], axis=1)
        Wt = _np.asarray(W)
    return Wt, _channel_sigma(amps, Wt, {1: _R23}, norm)
