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

from adonis.io import achilles_data_root
from adonis.channels.dcc.form_factors import M_PI_GEV  # GeV; we work in MeV here

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


def _pcm2(W, mM=138.5, mB=938.5):
    # ANL-code flux masses (ACHILLES MesonBaryonAmplitudes.hh:110-111 Mass_m[0]=138.5, Mass_b[0]=938.5),
    # NOT the physical PDG masses -- using mpip/mp here inflates the near-threshold piN normalization by
    # up to ~38% (matches the _MM_ANL/_MB_ANL convention already used for the eta/conversion grids below).
    # [audit 2026-07-20; see docs constants registry]
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


def dsigma_dOmega(W, cos_theta, cg={3: 1.0}, i=0, f=0):
    """piN -> piN differential cross section dsigma/dOmega(theta) [arb. shape] at fixed W,
    via the standard spin-non-flip f and spin-flip g partial-wave amplitudes:

        f(theta) = sum_L [ (L+1) a_{L+} + L a_{L-} ] P_L(cos)
        g(theta) = sum_L [ a_{L+} - a_{L-} ] P_L^1(cos)        (a_{L+/-}: J = L +/- 1/2)
        dsigma/dOmega = |f|^2 + |g|^2

    a_{L,J} is the isospin-combined amplitude sum_I cg_I A^I_{L,J}.  At the Delta (W~1232)
    P33 (L=1, J=3/2=L+1/2) dominates -> the classic 1 + 3 cos^2(theta) shape.

    (i, f) select the ANL channel table load_anl(i, f); the default (0, 0) is elastic piN, the
    ONLY case this f/g construction is validated for -- the kwargs exist so callers can share this
    one implementation, not to claim off-diagonal (i != f) support.
    """
    from numpy.polynomial.legendre import Legendre
    Wt, amps = load_anl(i, f)
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


# --- TOTAL piN -> {etaN, KLambda, KSigma} CONVERSION cross section ---------------------- #
# Faithful port of ACHILLES MesonBaryonAmplitudes (initIso CGcof + CalcCrossSectionW_grid):
# sigma(c -> f)(W) = pref(W) * sum_{L,J} (2J+1) | sum_I CG_I(c) CG_I(f) A^{0->F}_{L,J,I}(W) |^2
# with pref using the INITIAL channel "masses in ANL code" mM=138.5, mB=938.5 (ACHILLES
# Mass_m/Mass_b[0]) and the conversion total = sum over the OPEN final charge states with the
# same total I3.  These channels remove the pion (eta/K production) -- in the cascade they
# CONVERT the pion out of the pi+/pi0/pi- system.

_MM_ANL, _MB_ANL = 138.5, 938.5            # ACHILLES Mass_m[0], Mass_b[0] (ANL-code masses)
_C13, _C23 = np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)
# CG[(2*I3m, 2*I3b)] = (c_{1/2}, c_{3/2})  -- ACHILLES initIso chan 0 (piN), Condon-Shortley
_CG_PIN = {(+2, +1): (0.0, 1.0), (+2, -1): (_C23, _C13),
           (0, +1): (-_C13, _C23), (0, -1): (_C13, _C23),
           (-2, +1): (-_C23, _C13), (-2, -1): (0.0, 1.0)}
# KSigma final CGs: ACHILLES initIso chan 3 = piN CG with (meson<->baryon) swapped indices and
# a -1 on the I=1/2 row (mirrored AS CODED).  Keys: (2*I3_K, 2*I3_Sigma), I3_K in {+-1/2}, I3_S in {-1,0,1}.
_CG_KSIG = {(km, sb): (-_CG_PIN[(sb, km)][0], _CG_PIN[(sb, km)][1])
            for km in (+1, -1) for sb in (+2, 0, -2) if (sb, km) in _CG_PIN}


def _sigma_cf_masses(amps, Wt, cg_pair, mM, mB):
    """sigma(W) [mb] for one (charge-in -> charge-out) pair with EXPLICIT initial-channel PF masses.
    ACHILLES CalcCrossSectionW_grid uses mM=Mass_m[iMB_i], mB=Mass_b[iMB_i] -- the INITIAL channel
    (ANL-code) masses -- for ALL final channels of that initial state.  cg_pair = (c_half, c_three2)
    PRODUCTS of initial x final isospin CGs."""
    cg = {1: cg_pair[0], 3: cg_pair[1]}
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
    PF = (Wt ** 2 - mM ** 2 - mB ** 2) ** 2 - 4.0 * mM ** 2 * mB ** 2
    pref = np.where(PF > 0, HBARC ** 2 * 10.0 * 2.0 * np.pi * 4.0 * Wt ** 2 / np.clip(PF, 1e-9, None), 0.0)
    return pref * s


def _sigma_cf(amps, Wt, cg_pair):
    """piN-initial conversion sigma (PF uses the piN ANL-code masses Mass_m[0]/Mass_b[0])."""
    return _sigma_cf_masses(amps, Wt, cg_pair, _MM_ANL, _MB_ANL)


def conversion_sigma_grid():
    """W grid + total conversion sigma [mb] per piN charge channel, shape (3 pion, 2 nucleon, nW).
    Pion index 0=pi+ 1=pi0 2=pi-; nucleon 0=p 1=n (cascade convention).  Sums the open
    {etaN, KLambda, KSigma} final charge states at each total I3 (incoherent over finals,
    coherent over I within each final)."""
    out = None
    Wg = None
    for F, finals in ((1, "eta"), (2, "klam"), (3, "ksig")):
        Wt, amps = load_anl(0, F)
        if Wg is None:
            Wg = Wt
        elif not np.array_equal(Wt, Wg):                      # resample onto the first grid
            amps = np.stack([np.interp(Wg, Wt, amps[:, k], left=0, right=0)
                             for k in range(amps.shape[1])], axis=1)
        if out is None:
            out = np.zeros((3, 2, len(Wg)))
        for pi_idx, tm in ((0, +2), (1, 0), (2, -2)):
            for nuc_idx, tb in ((0, +1), (1, -1)):
                ci = _CG_PIN[(tm, tb)]
                I3tot = tm + tb                                # 2*I3 total
                if finals == "eta":                            # eta(I=0) + N: one final, I=1/2 only
                    if abs(I3tot) > 1:
                        continue
                    out[pi_idx, nuc_idx] += _sigma_cf(amps, Wg, (ci[0] * 1.0, 0.0))
                elif finals == "klam":                         # K(1/2) + Lambda(0): one final, I=1/2
                    if abs(I3tot) > 1:
                        continue
                    out[pi_idx, nuc_idx] += _sigma_cf(amps, Wg, (ci[0] * 1.0, 0.0))
                else:                                          # K(1/2) + Sigma(1): sum open finals
                    for km in (+1, -1):
                        sb = I3tot - km
                        if (km, sb) not in _CG_KSIG:
                            continue
                        cf = _CG_KSIG[(km, sb)]
                        out[pi_idx, nuc_idx] += _sigma_cf(amps, Wg, (ci[0] * cf[0], ci[1] * cf[1]))
    return Wg, out


# --- eta N INITIAL cross sections (the BACK-conversion path: ACHILLES propagates the eta produced by
# piN->etaN and lets it re-interact via GetAllCSW(eta_channel, W) -- etaN->etaN elastic and etaN->piN,
# which REGENERATES a pion.  eta is isoscalar => etaN is pure I=1/2 (initial CG = 1); the PF uses the
# INITIAL etaN ANL-code masses Mass_m[1]=548.0, Mass_b[1]=938.5). -------------------------------------- #
_MM_ETA, _MB_ETA = 548.0, 938.5            # ACHILLES Mass_m[1], Mass_b[1] (etaN initial, ANL-code masses)


def eta_production_sigma_grid():
    """W grid + piN -> etaN production sigma [mb] ONLY (the eta piece of the total conversion), shape
    (3 pion, 2 nucleon, nW).  This is the fraction of a pion conversion that produces an eta (which is
    then propagated & can back-convert), as opposed to the KLambda/KSigma finals (terminal).  Same
    piN-initial PF (Mass_m[0]/Mass_b[0]) as conversion_sigma_grid, restricted to the eta final."""
    Wt, amps = load_anl(0, 1)                                  # piN -> etaN amplitudes
    out = np.zeros((3, 2, len(Wt)))
    for pi_idx, tm in ((0, +2), (1, 0), (2, -2)):
        for nuc_idx, tb in ((0, +1), (1, -1)):
            if abs(tm + tb) > 1:                               # etaN is I=1/2 -> |2*I3| <= 1
                continue
            ci = _CG_PIN[(tm, tb)]
            out[pi_idx, nuc_idx] = _sigma_cf(amps, Wt, (ci[0] * 1.0, 0.0))
    return Wt, out


def eta_elastic_sigma_grid():
    """W grid + etaN -> etaN elastic sigma [mb].  Pure I=1/2 (initial & final CG = 1); identical for
    proton and neutron.  ANL_1-1 amplitudes, etaN-initial PF masses."""
    Wt, amps = load_anl(1, 1)
    return Wt, _sigma_cf_masses(amps, Wt, (1.0, 0.0), _MM_ETA, _MB_ETA)


def eta_backconv_sigma_grid():
    """W grid + etaN -> piN back-conversion sigma [mb], shape (2 nucleon, 3 out-pion).  nucleon 0=p 1=n;
    out-pion 0=pi+ 1=pi0 2=pi-.  Initial etaN is pure I=1/2 (CG=1); the outgoing piN carries its charge
    Clebsch (only the I=1/2 piece survives -> c_three_final is multiplied by the zero initial c_three).
    Charge conservation fixes the outgoing nucleon: 2*I3(N_out) = 2*I3(N_in) - 2*I3(pi_out).  ANL_1-0
    amplitudes, etaN-initial PF masses."""
    Wt, amps = load_anl(1, 0)
    out = np.zeros((2, 3, len(Wt)))
    for nuc_idx, tb in ((0, +1), (1, -1)):                     # incoming nucleon 2*I3 (p=+1, n=-1)
        for pi_idx, tm in ((0, +2), (1, 0), (2, -2)):          # outgoing pion 2*I3 (pi+=+2, pi0=0, pi-=-2)
            tb_out = tb - tm                                    # outgoing nucleon 2*I3 (I3(eta)=0)
            if tb_out not in (+1, -1):                          # charge-forbidden final state
                continue
            c_half_f = _CG_PIN[(tm, tb_out)][0]                # final piN I=1/2 Clebsch
            out[nuc_idx, pi_idx] = _sigma_cf_masses(amps, Wt, (c_half_f, 0.0), _MM_ETA, _MB_ETA)
    return Wt, out
