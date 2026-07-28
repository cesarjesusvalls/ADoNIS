"""Unified, jitted, differentiable intranuclear cascade (ADoNIS FSI).

ONE engine for the whole final-state cascade: pion + nucleon + secondary transport
over a shared nucleus, pooled (M=1 active slot + FIFO wait queue + persistent event
refill), differentiable via kind-1 reweighting, and jit-able end to end.

Merged verbatim (2026-07-26, branch cascade-unify) from the retired
cascade_full.py (pool orchestration) + cascade_discrete.py (step physics + reweight
math) + cascade_real.py (density/kinematics helpers) + nucleon_cascade.nn_elastic_sigma.
Cross-section source-of-truth stays in the imported libs (oset_xsec, interactions.meson_baryon_xsec,
nn_inelastic, absorption_modes). Public entry: cascade_nucleus (eager core) and
cascade_nucleus_jit (jitted). Config: CascadeConfig (DiscreteCascadeConfig = alias).
"""
from __future__ import annotations

import os
import gzip
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox
from adonis.fsi.interactions import meson_baryon_xsec
from adonis.fsi.absorption_modes import kernel_tables as _abs_kernel_tables
from adonis.fsi import nn_inelastic as nni
from adonis.constants import mp as _MP_PHYS, mn as _MN_PHYS

HBARC = ox.HBARC
M_N = ox.M_N
MB_TO_FM2 = 0.1                     # 1 mb = 0.1 fm^2
_PI_MASS = {211: ox.M_PIP, 111: ox.M_PI0, -211: ox.M_PIP}
_DENS = {}


def _read_density_file(name):
    p = Path(__file__).resolve().parents[2] / "data" / "nuclear" / name
    d = np.loadtxt(p, comments="#")
    return d[:, 0], d[:, 1]               # r [fm], rho [fm^-3]


def _load_density(name="c12_density.txt", name_n=None):
    """Proton & neutron number densities.  ACHILLES ALWAYS reads SEPARATE p/n densities
    (Nucleus.cc:36-80); for N=Z nuclei (C) name_n is name -> rho_n == rho_p.  rho_n is interpolated
    onto the proton radial grid so a single rgrid serves both species.  Radius = FIRST r where
    rho_proton < 1e-6 fm^-3 (ABSOLUTE minDensity, Nucleus.cc:49-51).  Returns (rgrid, rho_p, rho_n,
    radius).  For N=Z, rho_p == rho_n bitwise and rho_p+rho_n == 2*rho_p -> carbon is unchanged."""
    # numpy cache + per-call asarray: a jnp array first created inside a jit trace would
    # leak the tracer context to later traces (cf. meson_baryon_xsec._jax_grids_resolved).
    name_n = name_n or name
    key = (name, name_n)
    if key not in _DENS:
        r, rho_p = _read_density_file(name)
        if name_n == name:
            rho_n = rho_p
        else:
            rn, rhon0 = _read_density_file(name_n)
            rho_n = np.interp(r, rn, rhon0, left=rhon0[0], right=0.0)
        _below = r[rho_p < 1.0e-6]
        radius = float(_below.min()) if _below.size else float(r.max())
        _DENS[key] = (r, rho_p, rho_n, radius)
    r, rho_p, rho_n, radius = _DENS[key]
    return jnp.asarray(r), jnp.asarray(rho_p), jnp.asarray(rho_n), radius


def _rho_species(r, rgrid, rho):
    """Number density of ONE species at radius r [fm], interpolated; 0 beyond the grid.  Caller
    passes the proton OR neutron density array (rho_p / rho_n) -- they are equal for N=Z nuclei."""
    return jnp.interp(r, rgrid, rho, left=rho[0], right=0.0)


def _kf_local(rho_species):
    """Local Fermi momentum [MeV]: k_F = cbrt(rho * 3 pi^2) * hbarc (ACHILLES Local FG)."""
    return jnp.cbrt(jnp.clip(rho_species, 0.0, None) * 3.0 * jnp.pi ** 2) * HBARC


def _two_body_cm_scatter(p_pi, p_N, m_out_pi, key, cos_cm=None, m_recoil=None):
    """Elastic/charge-exchange piN -> pi'N' (or NN -> N'N') two-body kinematics; cos(theta_cm) is supplied
    (from the DCC angular distribution -- ACHILLES MesonBaryonInteraction::GenerateMomentum
    samples the partial-wave angular CDF, NOT isotropic) or isotropic if cos_cm is None.
    Returns the OUTGOING (m_out_pi) 4-momentum (E,px,py,pz) in the lab.  theta_cm is measured from
    the incoming-particle CM direction (Poincare z-axis), matching ACHILLES.
    m_recoil = the recoil particle's mass (default avg M_N); pass the PHYSICAL per-species nucleon mass
    so the 2->2 final-state masses match ACHILLES GenerateMomentum (ma/mb physical)."""
    P = p_pi + p_N                                  # total 4-momentum (lab)
    s = P[0] ** 2 - jnp.sum(P[1:] ** 2)
    sqrts = jnp.sqrt(jnp.clip(s, 1e-6, None))
    m1 = m_out_pi
    m2 = M_N if m_recoil is None else m_recoil
    E1 = (sqrts / 2.0) * (1.0 + (m1 ** 2 - m2 ** 2) / s)
    lam = jnp.sqrt(jnp.clip((s - m1 ** 2 - m2 ** 2) ** 2 - 4 * m1 ** 2 * m2 ** 2, 0.0, None))
    pf = lam / (2.0 * sqrts)
    ka, kb = jax.random.split(key)
    cth = (2.0 * jax.random.uniform(ka) - 1.0) if cos_cm is None else cos_cm
    sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0.0, None))
    phi = 2 * jnp.pi * jax.random.uniform(kb)
    # CM scatter axis = the incoming pion CM direction (theta measured from it), per ACHILLES
    beta = P[1:] / P[0]
    zaxis = _boost(p_pi, -beta)[1:]
    zhat = zaxis / jnp.clip(jnp.linalg.norm(zaxis), 1e-9, None)
    # build an orthonormal frame (zhat, e1, e2)
    ref = jnp.where(jnp.abs(zhat[2]) < 0.9, jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.]))
    e1 = jnp.cross(ref, zhat); e1 = e1 / jnp.clip(jnp.linalg.norm(e1), 1e-9, None)
    e2 = jnp.cross(zhat, e1)
    dir_cm = cth * zhat + sth * (jnp.cos(phi) * e1 + jnp.sin(phi) * e2)
    p1_cm = jnp.concatenate([E1[None], pf * dir_cm])
    return _boost(p1_cm, beta)


def _boost(p4, beta):
    """Lorentz boost of 4-vector p4 by velocity beta (boost from a frame moving at -beta)."""
    b2 = jnp.sum(beta ** 2)
    g = 1.0 / jnp.sqrt(jnp.clip(1 - b2, 1e-12, None))
    bp = jnp.sum(beta * p4[1:])
    E = g * (p4[0] + bp)
    fac = (g - 1.0) * jnp.where(b2 > 1e-12, bp / jnp.clip(b2, 1e-12, None), 0.0) + g * p4[0]
    vec = p4[1:] + fac * beta
    return jnp.concatenate([E[None], vec])


def _sample_fermi_nucleon(kf, key, mass=M_N):
    """Draw a nucleon 3-momentum uniformly in the Fermi sphere |p|<kf, on-shell.  `mass` is the
    PHYSICAL per-species nucleon mass (ACHILLES uses ParticleInfo(PID).Mass()); default = global avg."""
    kdir, kmag = jax.random.split(key)
    d = jax.random.normal(kdir, (3,)); d = d / jnp.linalg.norm(d)
    pmag = kf * jax.random.uniform(kmag) ** (1.0 / 3.0)
    p3 = d * pmag
    E = jnp.sqrt(mass ** 2 + pmag ** 2)
    return jnp.concatenate([E[None], p3])


# meson charge/species <-> index:  0:pi+ (211), 1:pi0 (111), 2:pi- (-211), 3:eta (221).
# Index 3 tags the eta produced by piN->etaN conversion, which the pool propagates as a meson (routed
# to _pion_step, species=PION) so it can back-convert etaN->piN and regenerate a pion (ACHILLES).
_M_ETA_PHYS = 547.862                                 # eta mass [MeV] (PDG; ACHILLES ParticleInfo)
_CH_PID = jnp.array([211, 111, -211, 221])
_CH_MASS = jnp.array([ox.M_PIP, ox.M_PI0, ox.M_PIP, _M_ETA_PHYS])
_QE_IN = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])      # index into pi+/0/- for QE_CHANNELS in
_QE_OUT = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])     # ... out

def sample_vertex(key, n, nucleus="c12_density.txt", density_n=None):
    """Sample n production vertices in the nucleus ~ rho(r) (radial pdf rho(r) r^2), as the
    pion's cascade starting point.  Uses the proton density grid (== total shape for N=Z)."""
    rgrid, rho, _rho_n, radius = _load_density(nucleus, density_n)
    rg = np.asarray(rgrid); rh = np.asarray(rho)
    rr = np.linspace(0.0, float(radius), 600)
    pdf = np.interp(rr, rg, rh, left=rh[0], right=0.0) * rr ** 2
    pdf = pdf / pdf.sum()
    kr, kd = jax.random.split(key)
    # inverse-CDF radial sample (detached)
    cdf = jnp.cumsum(jnp.asarray(pdf))
    u = jax.random.uniform(kr, (n,))
    idx = jnp.clip(jnp.searchsorted(cdf, u), 0, len(rr) - 1)
    rad = jnp.asarray(rr)[idx]
    dirs = jax.random.normal(kd, (n, 3)); dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs * rad[:, None]


M_N_GEV = M_N / 1000.0

def nn_elastic_sigma(sqrts_mev, same_iso, mn_gev=M_N_GEV):
    """NN elastic cross section [mb] (ACHILLES NNElastic, GiBUU).  sqrts in MeV; `same_iso`
    True for pp/nn, False for np.  Piecewise in p_lab [GeV].  `mn_gev` is the PER-PAIR average mass
    (mp+mp)/2 for pp, (mn+mn)/2 for nn, (mp+mn)/2 for pn -- ACHILLES NNElastic.cc:184 uses the actual
    pair mass in threshold/plab and the low-plab mn/threshold terms (default = global avg for callers
    that do not resolve the pair)."""
    sqrts = sqrts_mev / 1000.0                       # GeV
    mn = mn_gev
    thr = sqrts ** 2 - 4.0 * mn ** 2
    plab = jnp.where(thr > 0, sqrts / (2.0 * mn) * jnp.sqrt(jnp.clip(thr, 0.0, None)), 0.0)
    thr_safe = jnp.clip(thr, 1e-6, None)
    L = jnp.log(sqrts ** 2)

    def same():
        return jnp.where(plab < 0.425, 5.12 * mn / thr_safe + 1.67,
               jnp.where(plab < 0.8, 23.5 + 1000.0 * (plab - 0.7) ** 4,
               jnp.where(plab < 2.0, 1250.0 / (plab + 50.0) - 4.0 * (plab - 1.3) ** 2,
               jnp.where(plab < 6.0, 77.0 / (plab + 1.5),
                         11.84 - 1.617 * L + 0.1359 * L ** 2))))

    def diff():
        return jnp.where(plab < 0.525, 17.05 * mn / thr_safe - 6.83,
               jnp.where(plab < 0.8, 33.0 + 196.0 * jnp.abs(plab - 0.95) ** 2.5,
               jnp.where(plab < 2.0, 31.0 / jnp.sqrt(jnp.clip(plab, 1e-6, None)),
               jnp.where(plab < 6.0, 77.0 / (plab + 1.5),
                         11.84 - 1.617 * L + 0.1359 * L ** 2))))

    sig = jnp.where(same_iso, same(), diff())
    return jnp.where(thr > 0, jnp.clip(sig, 0.0, None), 0.0)


# pion-absorption proton-count distribution + partner species, indexed by ch*2+struck_p (ch 0:pi+
# 1:pi0 2:pi-; struck_p 1=proton).  Faithful ACHILLES isospin partition (Nucl.Phys. A568) -- replaces
# the old geometric nearest-partner pick that biased proton multiplicity for neutron-rich targets.
_ABS_W_NP, _ABS_PART_NP = _abs_kernel_tables()      # (6,3) numpy constants

M_N = ox.M_N                                          # average nucleon mass (Constant::mN = (mp+mn)/2)
from adonis.constants import mp as _MP_PHYS, mn as _MN_PHYS   # PHYSICAL per-species (ACHILLES particle 4-vec E)
HBARC = ox.HBARC
_CFG = {}


def _formation_zone(p_in, p_out):
    """ACHILLES Particle::SetFormationZone: fz = E_in * hbarc / |mN^2 - p_in.p_out|  [fm].
    p_in (n,4) = incoming nucleon momentum, p_out (n,4) = outgoing.  Forward scatters (p_in~p_out)
    -> |mN^2 - p_in.p_out| -> 0 -> large fz (free-streams out); wide scatters -> small fz."""
    dot4 = p_in[:, 0] * p_out[:, 0] - jnp.sum(p_in[:, 1:] * p_out[:, 1:], axis=1)
    return p_in[:, 0] * HBARC / jnp.clip(jnp.abs(M_N ** 2 - dot4), 1e-6, None)


def _load_qmc_configs(nmax=36000, name="QMC_configs.out.gz"):
    """Load A-nucleon configurations (positions [fm], isospin proton-mask, per-config weight).
    nmax=36000 = ALL configs in QMC_configs.out.gz (ACHILLES uses the full set); verified the
    transparency is unchanged vs the old 20000 cap (first-20k and full-36k have identical rms radius
    and both sample ∝ weight)."""
    # Cache as NUMPY (not jnp): a jnp array first created inside a jit trace would leak the
    # tracer context to later traces (cf. meson_baryon_xsec._jax_grids_resolved); asarray per-call is free.
    if name not in _CFG:
        from adonis.io import achilles_sibling_root
        path = achilles_sibling_root() / "data" / "configurations" / name   # single source: adonis.io
        with gzip.open(path, "rt") as f:
            # header: [A Nconfigs maxWgt minWgt] (ACHILLES Configuration.cc:30-37).  A is read HERE,
            # not hardcoded -> QMC (A=12, C) and RMF (A=40, Ar) share this parser unchanged.
            hdr = f.readline().split()
            A = int(hdr[0]); ncfg = int(hdr[1]); nread = min(nmax, ncfg)
            iso = np.zeros((nread, A), bool); pos = np.zeros((nread, A, 3)); wt = np.zeros(nread)
            for c in range(nread):
                for i in range(A):
                    t = f.readline().split()
                    iso[c, i] = float(t[0]) > 0
                    pos[c, i] = [float(t[1]), float(t[2]), float(t[3])]
                wt[c] = float(f.readline()); f.readline()
        _CFG[name] = (pos, iso, wt / wt.sum(), A)
    pos, iso, w, A = _CFG[name]
    return jnp.asarray(pos), jnp.asarray(iso), jnp.asarray(w), A


_KSLAB = 3                   # # of nearest in-slab nucleons whose cross sections are evaluated per step
                             # (fast_xsec).  sigma for non-top-K in-slab nucleons is forced 0.  ">=K in one
                             # 0.04 fm slab is ~never" (measured KSLAB=1==3 on n-C, 2026-07-26), so bit-exact.


def pion_branch_reweight(brec, sabs, sscat):
    """Kind-1 branching reweight from compressed walk records brec =
    (branch (n,K) int {0 scatter, 1 abs, 2 conversion}, sa, ss, si (n,K), n_hits (n,)).
    Per-hit likelihood ratio p_branch(theta)/p_branch(nominal) with
    p_abs = sabs*sa/D, p_scat = sscat*ss/D, p_conv = si/D, D = sabs*sa + sscat*ss + si
    (conversion sigma unscaled).  Pure in (sabs, sscat); == in-propagation w_fsi; reduces to
    the previous two-branch formula where si = 0."""
    bc, sa, ss, si, nh = brec
    valid = jnp.arange(sa.shape[1])[None, :] < nh[:, None]
    ss = jnp.clip(ss, 1e-6, None)
    D0 = sa + ss + si
    Dk = sabs * sa + sscat * ss + si
    num = jnp.where(bc == 1, sabs * sa, jnp.where(bc == 2, si, sscat * ss))
    den = jnp.where(bc == 1, sa, jnp.where(bc == 2, si, ss))
    br = (num / Dk) / jnp.clip(den / D0, 1e-12, None)
    return jnp.prod(jnp.where(valid, br, 1.0), axis=1)


def _match_dtype(a, g):
    """Promote the survival pair (a, g) to a common dtype BEFORE exponentiating.

    The bank stores `a` as float32 while `g` inherits float64 from the knob vector, so `exp(-a)` would be
    a float32 exp and `exp(-a/g)` a float64 one.  At nominal g == 1 exactly, so the two are mathematically
    identical -- yet they differ by the float32 rounding of exp, which breaks the reweight's NOMINAL
    IDENTITY by ~9e-4 per event (measured on the bank).  Exponentiate both in the same dtype."""
    dt = jnp.result_type(a, g)
    return jnp.asarray(a, dt), jnp.asarray(g, dt)


def _dense_prod(per, count):
    """Per-event product of a DENSE (n, K) slot factor, masking the unused tail slots."""
    valid = jnp.arange(per.shape[1])[None, :] < count[:, None]
    return jnp.prod(jnp.where(valid, per, 1.0), axis=1)


def _ragged_prod(per, eidx, n_events):
    """Per-event product of a RAGGED (M,) slot factor: exp(segment-sum of log).

    Every slot factor is a likelihood ratio, hence strictly positive, so the log is safe.  At nominal
    every factor is EXACTLY 1 -> log 0 -> segment-sum 0 -> exp(0) = 1, so the nominal identity stays
    bit-exact under this reduction (a plain scatter-multiply does not exist in XLA).

    The log-sum ACCUMULATES in the default float dtype (float64 under jax_enable_x64) even though the bank
    stores the slot sigmas as float32: summing ~2M logs in float32 would throw away precision the dense
    float32 product does not have to spend.  Cost is transient only -- nothing is stored at this width."""
    acc = jnp.result_type(float)                 # float64 when x64 is on, float32 otherwise
    lp = jnp.log(jnp.clip(per, 1e-300, None)).astype(acc)
    return jnp.exp(jnp.zeros(n_events, acc).at[eidx].add(lp))


# ---- PER-SLOT likelihood ratios: the physics, written ONCE.  Shape-agnostic -- the same function serves
# the dense in-engine record (n, K) and the ragged bank record (M,); only the reduction differs. --------- #
def pion_slot_factor(code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c,
                     s_abs, s_el, s_cex, s_conv):
    """Per-slot pion FSI likelihood ratio -- BRANCH x SURVIVAL.  One slot = one IN-SLAB CANDIDATE STEP
    (hit or not), the closest in-slab nucleon at that step.

    A sigma scale moves TWO things and the reweight must carry both (the nucleon record always did; the
    pion one carried only the first until 2026-07-14 -- see docs/logbook/info_content.md):
      (1) BRANCH, given an interaction:  s_realized * D0/D,   D0 = sa+ss+si,
          D = s_abs*sa + s_el*ss_el + s_cex*(ss-ss_el) + s_conv*si   [at the STRUCK candidate]
      (2) SURVIVAL -- WHETHER it interacts.  The walk draws the hit with prob = exp(-a) per candidate
          (_pion_step), and sigma_tot -> g*sigma_tot maps a -> a/g, so with g = D_c/D0_c at the CLOSEST
          candidate:   hit: pk/p0    no-hit: (1-pk)/(1-p0),    p0 = exp(-a), pk = exp(-a/g).
    Under a COMMON rescale s the branch factor is 1 but the survival factor is NOT -- which is the whole
    point (a common rescale changes the mean free path).  == 1 at all-nominal (g = 1 -> pk = p0)."""
    # On a NO-HIT slot the branch stats are taken at j = argmin over an all-inf metric -> meaningless (and
    # possibly non-finite).  jnp.where picks the right VALUE, but a non-finite untaken branch still poisons
    # the reverse-mode GRADIENT (nan * 0 = nan), so neutralize the branch inputs off-hit up front.
    hitf = hh.astype(bool)
    sa = jnp.where(hitf, sa, 1.0); ss_el = jnp.where(hitf, ss_el, 1.0)
    ss = jnp.where(hitf, ss, 1.0); si = jnp.where(hitf, si, 0.0)

    def _sigma_ratio(sa_, ss_el_, ss_, si_):
        """D/D0 = sum_i s_i f_i, written as 1 + sum_i (s_i - 1) f_i.

        Algebraically identical, but EXACTLY 1 at nominal in ANY precision (every (s_i - 1) is 0),
        whereas D/D0 is only 1 up to the rounding of D and D0.  The bank stores these sigmas as float32
        and exp(-a/g) amplifies a g that is off by even 1 ulp."""
        D0 = jnp.clip(sa_ + ss_ + si_, 1e-12, None)
        fcex = jnp.clip(ss_ - ss_el_, 0.0, None) / D0
        return (1.0 + (s_abs - 1.0) * (sa_ / D0) + (s_el - 1.0) * (ss_el_ / D0)
                + (s_cex - 1.0) * fcex + (s_conv - 1.0) * (si_ / D0))

    s_real = jnp.where(code == 0, s_el, jnp.where(code == 1, s_cex, jnp.where(code == 2, s_abs, s_conv)))
    branch = s_real / jnp.clip(_sigma_ratio(sa, ss_el, ss, si), 1e-12, None)
    g = jnp.clip(_sigma_ratio(sa_c, ss_el_c, ss_c, si_c), 1e-6, None)
    a, g = _match_dtype(a, g)                    # same dtype for both exps (see _match_dtype)
    p0 = jnp.clip(jnp.exp(-a), 1e-6, 1.0 - 1e-6)
    pk = jnp.clip(jnp.exp(-a / g), 1e-6, 1.0 - 1e-6)
    surv = jnp.where(hitf, pk / p0, (1.0 - pk) / (1.0 - p0))
    return jnp.where(hitf, surv * branch, surv)


def nucleon_slot_factor(hh, a_nom, iso, finel, inel, s_el, s_inel):
    """Per-slot nucleon FSI likelihood ratio.  One slot = one in-slab candidate step; iso {0 pp,1 pn,2 nn};
    finel = sigma_in/sigma_tot there.  sigma_tot(s) = g*sigma_tot_0 with
      g = s_el[iso]*(1-finel) + s_inel[iso]*finel,
    so a -> a/g; a hit carries the Gaussian ratio exp(-a/g)/exp(-a) times the el/inel sub-branch
    s_realized/g, a no-hit carries the complement (1-pk)/(1-p0).  == 1 at all-nominal."""
    se = jnp.asarray(s_el); si = jnp.asarray(s_inel)
    se_i = se[iso]; si_i = si[iso]                                # per-candidate per-iso scales
    # 1 + (se-1)(1-finel) + (si-1)finel: EXACTLY 1 at nominal in any precision (see pion _sigma_ratio).
    g = jnp.clip(1.0 + (se_i - 1.0) * (1.0 - finel) + (si_i - 1.0) * finel, 1e-6, None)
    a_nom, g = _match_dtype(a_nom, g)
    p0 = jnp.clip(jnp.exp(-a_nom), 1e-6, 1.0 - 1e-6)
    pk = jnp.clip(jnp.exp(-a_nom / g), 1e-6, 1.0 - 1e-6)
    s_real = jnp.where(inel, si_i, se_i)
    hit_f = (pk / p0) * (s_real / g)                             # hit: Gaussian ratio x el/inel sub-branch
    return jnp.where(hh, hit_f, (1.0 - pk) / (1.0 - p0))


def nncex_slot_factor(hh, iso, inel, swap, f_cex):
    """Per-slot NN-elastic charge-exchange FRACTION ratio (ACHILLES value 0.5).  Only pn ELASTIC hits carry
    a meaningful swap: f_cex/0.5 if swapped else (1-f_cex)/0.5.  pp/nn swaps are no-ops (same species)."""
    pn_el = hh & (~inel) & (iso == 1)
    return jnp.where(pn_el, jnp.where(swap, f_cex / 0.5, (1.0 - f_cex) / 0.5), 1.0)


# ---- DENSE wrappers (in-engine record, (n, K) + per-event count) ------------------------------------- #
def fsi_pion_reweight(brec, s_abs, s_el, s_cex, s_conv):
    """brec = (code4, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c, np_ (n,)).  See pion_slot_factor."""
    code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c, np_ = brec
    return _dense_prod(pion_slot_factor(code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c,
                                        s_abs, s_el, s_cex, s_conv), np_)


def fsi_nucleon_reweight(srec, s_el, s_inel):
    """srec = (hh, a_nom, iso, finel, inel, ns (n,)).  See nucleon_slot_factor."""
    hh, a_nom, iso, finel, inel, ns = srec
    return _dense_prod(nucleon_slot_factor(hh, a_nom, iso, finel, inel, s_el, s_inel), ns)


def fsi_nncex_reweight(srec, f_cex):
    """srec = (hh, iso, inel, swap, ns (n,)).  See nncex_slot_factor."""
    hh, iso, inel, swap, ns = srec
    return _dense_prod(nncex_slot_factor(hh, iso, inel, swap, f_cex), ns)


# ---- RAGGED wrappers (bank record: flat (M,) slots + per-slot event index) ---------------------------- #
def fsi_pion_reweight_flat(brec, eidx, n_events, s_abs, s_el, s_cex, s_conv):
    code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c = brec
    return _ragged_prod(pion_slot_factor(code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c,
                                         s_abs, s_el, s_cex, s_conv), eidx, n_events)


def fsi_nucleon_reweight_flat(srec, eidx, n_events, s_el, s_inel):
    hh, a_nom, iso, finel, inel = srec
    return _ragged_prod(nucleon_slot_factor(hh, a_nom, iso, finel, inel, s_el, s_inel), eidx, n_events)


def fsi_nncex_reweight_flat(srec, eidx, n_events, f_cex):
    hh, iso, inel, swap = srec
    return _ragged_prod(nncex_slot_factor(hh, iso, inel, swap, f_cex), eidx, n_events)


def nucleon_scat_reweight(srec, sscat):
    """Kind-1 sigma_scatter reweight from compressed nucleon-walk records srec =
    (hit (n,K), a_nom (n,K), n_slab (n,)) with a_nom = pi b^2/(sigma fm^2) of the closest
    in-slab nucleon at each candidate step.  Pure in sscat; bit-exact equal to the
    in-propagation w_scat."""
    hh, a_nom, ns = srec
    valid = jnp.arange(a_nom.shape[1])[None, :] < ns[:, None]
    p_nom = jnp.clip(jnp.exp(-a_nom), 1e-6, 1.0 - 1e-6)
    p_knb = jnp.clip(jnp.exp(-a_nom / sscat), 1e-6, 1.0 - 1e-6)
    br = jnp.where(valid, jnp.where(hh, p_knb / p_nom, (1.0 - p_knb) / (1.0 - p_nom)), 1.0)
    return jnp.prod(br, axis=1)


@dataclass(frozen=True)
class CascadeConfig:
    nucleus: str = "c12_density.txt"       # proton density file (data/nuclear/)
    density_n: str = "c12_density.txt"     # neutron density file (= nucleus for N=Z nuclei, e.g. C)
    configs: str = "QMC_configs.out.gz"    # nucleon configuration file (QMC/RMF; A read from header)
    step: float = 0.05
    max_steps: int = 100000  # ABSOLUTE per-particle/per-event step ceiling (ACHILLES cMaxSteps = 100000).
                             # NOT a physics knob: the cascade terminates via escape/capture/absorption/
                             # path_budget_R far below this.  Reaching it = a runaway particle, which the
                             # pool RAISES on (run_cascade_pool _raise_if_runaway) rather than silently
                             # truncating.  Keep == cascade_full._HARD_STEPS.
    path_budget_R: float = 20.0  # RUNAWAY BACKSTOP (was 3.0): drop a particle once its accumulated path
                             # length exceeds path_budget_R * nuclear_radius (lpath >= R_budget).  ACHILLES
                             # has NO path budget (only geometric escape/capture + the 100k-step ceiling),
                             # so this is an ADoNIS-only cap with no counterpart.  At 3.0 it fired on
                             # ~0.009% of particles -- but those are exactly the slow, tightly random-
                             # walking tracks that ACHILLES lets run to escape/capture, so a too-tight
                             # budget can silently clip legitimate low-energy trajectories (and, being a
                             # silent kill with no fate, mis-count the primary as non-reacted).  20R is far
                             # beyond any real escaping/capturing track (net escape needs ~1R), so it
                             # changes compute negligibly (particles terminate physically first) while
                             # making the clipped fraction ~0.  Scheme-independent (distance/time-sync).
    seed: int = 0
    time_step: bool = False  # stepping clock: False = fixed-DISTANCE march (every particle advances
                             # `step` fm/global-step -> distance-synchronized).  True = fixed-TIME march
                             # mirroring ACHILLES AdaptiveStep: `step` is read as Dt and every particle
                             # advances `beta*step` fm/global-step (its own beta), so fast tracks outrun
                             # slow ones -> time-synchronized.  Decoupled (own beta, no max_beta) so
                             # P-invariance holds; fz then decrements by the shared Dt (=step), keeping
                             # the fz-expiry distance beta*fz invariant.  See cascade_subcascade_sequencing.md.
    # interaction probability is ALWAYS the Gaussian model exp(-pi b^2/sigma) -- the single,
    # differentiable interaction-probability law (the non-differentiable Cylinder hard-disk and the
    # unreliable ACHILLES "Pion" model were removed in the pool-unification cleanup).
    fast_xsec: bool = True   # evaluate Oset/DCC cross sections only for the K nearest in-slab nucleons
                             # (scatter back into the (n,A) grid); bit-exact, ~4x cheaper per step.
    pauli: bool = True       # Pauli-block the outgoing nucleon(s) of scatter/absorption (ACHILLES
                             # FinalizeMomentum); set False for ablation (no-blocking) studies.
    algo: str = "step"       # "step": fixed-step Glauber march (reference).  "interaction": jump
                             # directly to the next interaction (same probability model, ~20x fewer
                             # iterations).  Statistically equivalent; validated against "step".
    nn_inelastic: bool = True  # NN -> N Delta -> N N pi in the NUCLEON cascade (ACHILLES
                             # NucleonNucleon GiBUU ResonanceMode: Decay; adonis/fsi/nn_inelastic).
                             # Degrades fast nucleons and CREATES a pion (meson-veto relevant).
                             # False = the previous elastic-only walk (bit-exact).
    early_exit: bool = True  # while_loop walk that stops once NO particle can interact again
                             # (dead, or outside the radius moving outward = inert).  BIT-EXACT in
                             # every returned output (same per-step keys; skipped steps are
                             # identity), gated in tests/test_cascade_early_exit.py.  max_steps
                             # then acts as a pure safety bound.  False = the reference lax.scan.
    track_steps: bool = False  # MC-truth: also stack the per-step (pos, p4, alive) trajectory from the
                             # scan -> returns traj for viz/diagnostics (adonis/fsi/tracking).  Forces the
                             # reference scan (not early_exit).  DEFAULT OFF -> production path unchanged.
    engine: str = "pool"     # cascade structure: "pool" = single per-step-reconciled particle stack
                             # (DEFAULT, validated single core); "bfs" = legacy generation-synchronized; true step-order
                             # consumption; see docs/logbook/cascade_pool_engine.md).  WIP behind switch.
    recap_ke: float = 10.0   # escaping nucleon with KE < recap_ke [MeV] is RECAPTURED (set to rest).
                             # FAITHFUL to ACHILLES Cascade::Escaped (src/Achilles/Cascade.cc, called
                             # every step, UNGATED by PotentialProp): `constexpr double potential = 10.0;
                             # energy = E - mN - potential; if(|pos|>radius){ KE<10 -> captured }`.  This
                             # 10 MeV optical-potential capture is SEPARATE from (and always on, unlike)
                             # the PotentialProp:True Hamiltonian capture.  Mirrors ACHILLES's hard-coded
                             # 10.0 -> keep in sync with it; set 0.0 only for no-capture ablations.


def sample_nucleons(key, n, cfg: CascadeConfig):
    """Pick n configurations ~ weight; assign each nucleon an isotropic local-Fermi-gas momentum.
    Returns npos (n,A,3), nmom (n,A,4), nisp (n,A) proton-mask."""
    pos, iso, w, A = _load_qmc_configs(name=cfg.configs)
    rgrid, rhoP, rhoN, _ = _load_density(cfg.nucleus, cfg.density_n)
    kc, kd, km = jax.random.split(key, 3)
    idx = jax.random.choice(kc, pos.shape[0], (n,), p=w)
    npos = pos[idx]; nisp = iso[idx]
    # D10: per-draw random Euler rotation of the nucleon configuration (ACHILLES
    # DensityConfiguration::GetConfiguration, Configuration.cc:76-79: three angles ~U(0,2pi) with the
    # SECOND halved to [0,pi), applied via ThreeVector::Rotate's ZXZ matrix, ThreeVector.cc:31-40).
    # The density is spherically symmetric so r (hence k_F) is unchanged, but the FIXED-+z beam sees a
    # different nucleon arrangement per event -> orientation is no longer frozen to the config library's
    # stored frame.  Independent RNG stream (fold_in) so config/momentum draws above stay bit-identical.
    _ke = jax.random.fold_in(key, 777)
    _ang = jax.random.uniform(_ke, (n, 3)) * (2 * jnp.pi)
    _a, _b, _g = _ang[:, 0], _ang[:, 1] / 2.0, _ang[:, 2]      # beta halved -> [0,pi)
    c1, s1 = jnp.cos(_a), jnp.sin(_a); c2, s2 = jnp.cos(_b), jnp.sin(_b); c3, s3 = jnp.cos(_g), jnp.sin(_g)
    _R = jnp.stack([                                            # ZXZ matrix, ThreeVector::Rotate exactly
        jnp.stack([c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2], axis=1),
        jnp.stack([c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2], axis=1),
        jnp.stack([s2 * s3, c3 * s2, c2], axis=1)], axis=1)     # (n,3,3)
    npos = jnp.einsum('eij,eaj->eai', _R, npos)
    r = jnp.linalg.norm(npos, axis=2)
    # ACHILLES Local FG: PER-SPECIES k_F -- protons from rho_p, neutrons from rho_n (Nucleus.cc:212-238).
    # For N=Z (carbon) rho_p == rho_n bitwise -> unchanged.
    kf = _kf_local(jnp.where(nisp, _rho_species(r, rgrid, rhoP), _rho_species(r, rgrid, rhoN)))
    d = jax.random.normal(kd, (n, A, 3)); d = d / jnp.linalg.norm(d, axis=2, keepdims=True)
    pm = kf * jax.random.uniform(km, (n, A)) ** (1.0 / 3.0)
    m_sp = jnp.where(nisp, _MP_PHYS, _MN_PHYS)               # PHYSICAL per-species (ACHILLES Nucleus.cc:145)
    p3 = d * pm[:, :, None]; E = jnp.sqrt(m_sp ** 2 + pm ** 2)
    nmom = jnp.concatenate([E[:, :, None], p3], axis=2)
    return npos, nmom, nisp


# --- per-event RNG (persistent-refill engine) -------------------------------------------------------
# The step physics is keyed PER EVENT so a refilled event draws the SAME randoms regardless of which
# working-set slot / global step processes it (docs/logbook/cascade_persistent_refill_plan.md).  `key`
# into _nucleon_step/_pion_step is therefore an (n,2) array (one PRNG key per event), not a shared (2,)
# key.  These helpers vmap the per-event draws; distributions are unchanged (each event gets an
# independent stream), only WHICH draws each event sees differs from the old shared-key scheme.
def _ev_split(keys, k):
    """Per-event split: keys (n,2) -> (n,k,2)."""
    return jax.vmap(lambda key: jax.random.split(key, k))(keys)


def _ev_uniform(keys, shape=()):
    """Per-event uniform: keys (n,2) -> (n,*shape)."""
    return jax.vmap(lambda key: jax.random.uniform(key, shape))(keys)


def _ev_fold_uniform(keys, data, shape=()):
    """Per-event fold_in(data) then uniform: keys (n,2) -> (n,*shape)."""
    return jax.vmap(lambda key: jax.random.uniform(jax.random.fold_in(key, data), shape))(keys)


def _nucleon_step(p4, pos, dhat, fz, isp, alive, npos, nmom, nisp, consumed,
                  rgrid, rhoP, rhoN, radius, cfg, key, dt_evt=None, is_beam=None):
    """ONE step of the NUCLEON cascade for one particle per event (n,) -- the per-step physics of
    `_propagate_nucleon_discrete.body` (escape/recapture, formation zone, in-slab geometry, elastic
    scatter + per-species Pauli, NN->NDelta->NN'pi inelastic + channel charges), re-expressed for the
    POOLED engine: the knockout (recoil | inelastic 2nd nucleon) and the created pion are returned as
    IMMEDIATE spawns (no best_ko top-K deferral), and the updated `consumed` mask is returned for
    slot-serialized depletion.  RNG usage matches body exactly (split(key,3) + fold_in(sk,101..108)),
    so iterating this with the same per-step keys reproduces the bfs leading trajectory bit-for-bit.
    Returns: (p4', pos', dhat', fz', alive'), terminal, recap, do, (ko4,kopos,kofz,koisp,koal),
             (pi4,pipos,pifz,pich,pial), consumed', (has_hit, perp2_c, sig_c)."""
    n, A = nisp.shape; ar = jnp.arange(n)
    # D1: ACHILLES's Escaped() (Cascade.cc:640) is a PURE position test -- `Position().Magnitude2() >
    # radius^2` -- with NO directional gate.  We used to require `& outward` (momentum pointing away),
    # which retired strictly fewer particles than ACHILLES: one beyond the radius but momentarily moving
    # inward/tangentially (the surface competition zone for a created pion) stayed interaction-eligible
    # here while ACHILLES had already counted it escaped, giving extra reabsorption chances.
    esc_sphere = jnp.linalg.norm(pos, axis=1) > radius
    # ACHILLES external_test beam (is_beam == ParticleStatus::external_test, the un-scattered CrossSection
    # beam) escapes via the z>=radius PLANE (Cascade.cc:632); everything else -- scattered primary,
    # knockouts/secondaries, all RES/QE -- escapes via the sphere.  The sphere trips at z=sqrt(R^2-b^2)<R,
    # cutting the beam path short for impact parameter b>0 and missing distant large-b candidates (the
    # pn-channel first-hit deficit, Deviation 2).
    if is_beam is not None:
        reached = jnp.where(is_beam, pos[:, 2] >= radius, esc_sphere)   # reached the escape boundary
    else:
        reached = esc_sphere
    # D12 + capture-as-escape-partition.  ACHILLES Cascade::Escaped (Cascade.cc:640-649, every step,
    # ungated by PotentialProp): a nucleon reaching the boundary with KE < 10 MeV is CAPTURED (bound,
    # status 26) -- NOT a final-state particle -- while everything else becomes final_state (status 1).
    # We PARTITION the boundary-reachers into `escaping` (emitted) and `captured` (bound) HERE, at the
    # source, and hand back `escaping` = emitted-to-final-state ONLY.  Invariant: a captured nucleon is
    # never in `escaping`, so every downstream consumer of the escape flag (the pool's terminal/output
    # collection, the fate latch, diagnostics) is correct BY CONSTRUCTION -- there is no separate
    # "recapture" flag that a caller must remember to apply.  (The old code returned `escaping` INCLUDING
    # captured plus a side `recap` flag; the pool ignored `recap` and emitted captured protons at rest ->
    # the P(n_p=0) deficit.  Folding capture into the partition removes that whole bug class.)
    # CRITICAL mass convention: E uses the PHYSICAL per-species mass (neutron mn=939.565), while the
    # subtracted threshold uses the AVERAGE mN (Constant::mN=938.919); ADoNIS p4[:,0] carries avg M_N, so
    # recompute E from |p| with the physical species mass (using avg for E over-captures neutrons in
    # |p| in (132.9, 137.4) MeV -> the <137 MeV first-bin deficit).
    _e_phys = jnp.sqrt(jnp.where(isp, _MP_PHYS, _MN_PHYS) ** 2 + jnp.sum(p4[:, 1:] ** 2, axis=1))
    captured = reached & ((_e_phys - M_N) < cfg.recap_ke)      # KE < recap_ke -> bound, not emitted
    escaping = reached & ~captured                             # emitted to the final state ONLY
    alive = alive & ~reached                                   # emitted AND captured both leave the cascade
    beta = jnp.linalg.norm(p4[:, 1:], axis=1) / jnp.clip(p4[:, 0], 1e-9, None)
    # STEPPING CLOCK: distance-sync -> every particle sweeps `step` fm/step, fz -= step/beta (proper time).
    # time-sync (ACHILLES AdaptiveStep) -> the per-event time step dt_evt = step/beta_max (beta_max over the
    # event's alive particles, supplied by the pool) sets the clock: sweep beta*dt_evt fm/step (fast tracks
    # outrun slow ones; a particle ALONE has beta_max=beta -> sweeps `step`, never freezes), fz -= dt_evt.
    # Both schemes keep the fz-expiry distance = beta*fz invariant.
    if cfg.time_step:
        _dt = jnp.full_like(beta, cfg.step) if dt_evt is None else dt_evt
        _dstep = beta * _dt                           # distance swept this step = beta * Dt
        timeStep = _dt                                # fz decrement = shared per-event Dt
    else:
        _dstep = jnp.full_like(beta, cfg.step)        # fixed distance
        timeStep = cfg.step / jnp.clip(beta, 1e-6, None)
    can_int = fz <= 0.0
    rel = npos - pos[:, None, :]
    par = jnp.sum(rel * dhat[:, None, :], axis=2)
    perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
    in_slab = (par > 0) & (par <= _dstep[:, None]) & (~consumed) & alive[:, None]
    Pp = p4[:, None, :] + nmom
    s = Pp[:, :, 0] ** 2 - jnp.sum(Pp[:, :, 1:] ** 2, axis=2)
    same_iso = isp[:, None] == nisp
    # ACHILLES NNElastic.cc:184 uses the PER-PAIR average physical mass (mp for pp, mn for nn, avg for
    # pn) in threshold/plab + the low-plab mn/threshold terms -- not the global average.
    _m1_mev = jnp.where(isp, _MP_PHYS, _MN_PHYS)[:, None]           # (n,1) beam nucleon PHYSICAL mass
    _m2_mev = jnp.where(nisp, _MP_PHYS, _MN_PHYS)                   # (n,A) struck nucleon PHYSICAL mass
    _m_pair_gev = 0.5 * (_m1_mev + _m2_mev) / 1000.0
    # sqrts floor: PHYSICAL per-pair threshold (m1+m2)^2 so near-threshold thr can reach 0 (matching
    # ACHILLES NNElastic.cc:185-186), NOT pinned above the avg-mass floor (which caps the divergent low-p
    # pp sigma: 2*mN=1877.84 > 2*mp=1876.54).  [audit 2026-07-20; see docs constants registry]
    sqrts = jnp.sqrt(jnp.clip(s, (_m1_mev + _m2_mev) ** 2, None))
    sig_el = jnp.clip(nn_elastic_sigma(sqrts, same_iso, _m_pair_gev), 0.0, None)
    if cfg.nn_inelastic:
        # incoming NN CM momentum: exact Kallen with PHYSICAL per-pair masses (= ACHILLES p1CM boost,
        # NucleonNucleon.cc:57-59), NOT the equal-avg-mass approximation s/4 - M_N^2.
        _lam_in = (s - (_m1_mev + _m2_mev) ** 2) * (s - (_m1_mev - _m2_mev) ** 2)
        pcm = jnp.sqrt(jnp.clip(_lam_in, 0.0, None)) / (2.0 * sqrts) / 1000.0
        sig_in = jnp.clip(nni.sigma_nn_ndelta(sqrts / 1000.0, pcm, same_iso), 0.0, None)
    else:
        sig_in = jnp.zeros_like(sig_el)
    sig = sig_el + sig_in
    # Gaussian interaction probability (the single, differentiable model)
    prob = jnp.where(in_slab, jnp.exp(-jnp.pi * perp2 / jnp.clip(sig * MB_TO_FM2, 1e-12, None)), 0.0)
    _ks3 = _ev_split(key, 3); sk, ku, ks = _ks3[:, 0], _ks3[:, 1], _ks3[:, 2]   # per-event keys (n,2)
    passes = in_slab & (_ev_uniform(ku, (A,)) < prob)
    # ACHILLES Cascade::Interacted (Cascade.cc:759-786) walks candidates in ASCENDING IMPACT-PARAMETER b^2
    # order -- AllowedInteractions sorts by perp^2 via Project()->Magnitude2() (Cascade.cc:718-726) +
    # sortPairSecond (Utilities.cc:32) -- and returns the FIRST that passes its Gaussian roll.  With the
    # per-nucleon rolls fixed, "first passer in b^2 order" == "min-PERP2 among passers".  (A 2026-07-26
    # change to min-`par`/along-path MISREAD the ACHILLES sort key as path-order -- it is impact-parameter
    # order -- and was a regression biasing toward softer, larger-b recoils; reverted here.)
    big = jnp.where(passes, perp2, jnp.inf)        # ACHILLES: smallest-impact-parameter passer
    j = jnp.argmin(big, axis=1)
    has_hit = jnp.isfinite(big[ar, j]) & alive & can_int
    perp2_is = jnp.where(in_slab, perp2, jnp.inf); cidx = jnp.argmin(perp2_is, axis=1)
    has_slab = jnp.any(in_slab, axis=1)
    perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6); sig_c = sig[ar, cidx]
    # GRANULAR nucleon-FSI record: pair-isospin of the closest in-slab candidate {0 pp,1 pn,2 nn} and the
    # inelastic fraction finel = sig_in/sig_tot there -> lets fsi_nucleon_reweight scale per-iso elastic and
    # inelastic NN sigma independently (s_NN_elastic{pp,pn,nn}, s_NN_inelastic{pp,pn,nn}).
    _nisp_c = nisp[ar, cidx]
    iso_c = jnp.where(isp & _nisp_c, 0, jnp.where((~isp) & (~_nisp_c), 2, 1)).astype(jnp.int32)
    finel_c = jnp.clip(sig_in[ar, cidx] / jnp.clip(sig_c, 1e-12, None), 0.0, 1.0)
    pN_j = nmom[ar, j]
    rnuc = jnp.linalg.norm(npos, axis=2)
    kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP), _rho_species(rnuc, rgrid, rhoN)))
    kf_j = kf_n[ar, j]                                            # recoil (struck nucleon) k_F at the struck vertex
    _rnuc_j = rnuc[ar, j]
    kf_p_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoP))        # (charge-exchange recoil uses the struck vertex)
    kf_n_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoN))
    # LEADING outgoing Pauli k_F: evaluate at the LEADING's OWN position (|pos|), NOT the struck nucleon's
    # position -- ACHILLES PauliBlocking(paOut) uses k_F at the outgoing's own position.  For large-sigma
    # (slow-proton, near-threshold) scatters the struck nucleon sits up to ~the impact parameter (~2 fm)
    # away at a different density, so using its k_F leaked sub-k_F leading outgoing -> slow-proton
    # over-interaction (5sigma at 125-250 MeV vs ACHILLES).
    _r_lead = jnp.linalg.norm(pos, axis=1)
    _kfp_l = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_l = _kf_local(_rho_species(_r_lead, rgrid, rhoN))

    # ACHILLES NN ELASTIC CHARGE EXCHANGE: the two outgoing are {id1,id2} OR the id-SWAPPED {id2,id1}, each
    # at HALF the elastic sigma (NucleonNucleon.cc:62-65).  a-role = p_out (continues in the slot),
    # b-role = recoil; the swap exchanges their isospin (-> mass, Pauli k_F species, charge).  No-op for
    # pp/nn (isp == struck).  Per-event coin (fold 109), independent of the other channel draws.
    _swap_cx = _ev_fold_uniform(sk, 109) < 0.5
    _struck_isp = nisp[ar, j]
    lead_isp_out = jnp.where(_swap_cx, _struck_isp, isp)         # continuing nucleon's OUTGOING isospin
    rec_isp_out = jnp.where(_swap_cx, isp, _struck_isp)          # recoil's OUTGOING isospin
    kf_lead = jnp.where(lead_isp_out, _kfp_l, _kfn_l)            # leading Pauli k_F (its own position, swapped species)
    kf_rec = jnp.where(rec_isp_out, kf_p_j, kf_n_j)             # recoil Pauli k_F (struck vertex, swapped species)
    # 2->2 final-state masses PHYSICAL per-species (ACHILLES GenerateMomentum ma/mb), swap-aware.
    m_lead_phys = jnp.where(lead_isp_out, _MP_PHYS, _MN_PHYS)
    m_struck_phys = jnp.where(rec_isp_out, _MP_PHYS, _MN_PHYS)

    def scat_one(p_lead, pN_i, kf_out, kf_rec_, m1, m2, k):
        p_o = _two_body_cm_scatter(p_lead, pN_i, m1, k, m_recoil=m2)
        p_r = (p_lead + pN_i) - p_o
        return p_o, (jnp.linalg.norm(p_o[1:]) < kf_out) | (jnp.linalg.norm(p_r[1:]) < kf_rec_)
    p_out, blocked = jax.vmap(scat_one)(p4, pN_j, kf_lead, kf_rec, m_lead_phys, m_struck_phys, ks)
    if not cfg.pauli:
        blocked = blocked & False
    sig_in_j = sig_in[ar, j]; sig_el_j = sig_el[ar, j]
    u_br = _ev_fold_uniform(sk, 101)
    chose_inel = has_hit & (u_br < sig_in_j / jnp.clip(sig_el_j + sig_in_j, 1e-12, None))
    Pj = p4 + pN_j
    _mNb = jnp.where(isp, _MP_PHYS, _MN_PHYS)                       # beam nucleon PHYSICAL mass
    _mNs = jnp.where(nisp[ar, j], _MP_PHYS, _MN_PHYS)               # struck nucleon PHYSICAL mass
    rs_j = jnp.sqrt(jnp.clip(Pj[:, 0] ** 2 - jnp.sum(Pj[:, 1:] ** 2, axis=1), (_mNb + _mNs) ** 2, None))
    u_m = _ev_fold_uniform(sk, 102)
    _m_d_raw = nni.sample_delta_mass(rs_j / 1000.0, u_m) * 1000.0   # Delta-mass clip deferred (needs recoil mass)
    cth1 = 2 * _ev_fold_uniform(sk, 103) - 1.0
    phi1 = 2 * jnp.pi * _ev_fold_uniform(sk, 104)
    # D6: Delta->N pi decay follows the AngularMom:2 law w(cos) proportional to (1+3cos^2)/4, sampled by
    # ACHILLES's analytic inverse-CDF (DecayHandler.cc:126-131).  cbrt is the REAL cube root; the sqrt
    # argument 7-27r+27r^2 has negative discriminant so is always positive.  cth2 in [-1,1] (clip the
    # ~1e-2 float overshoot at the endpoints), symmetric, cth2(r=0.5)=0.  The NN->NDelta PRODUCTION
    # angle cth1 stays isotropic (NucleonNucleon.cc:129 cos_cms = 2r-1).
    _r2 = _ev_fold_uniform(sk, 105)
    _term2 = jnp.cbrt(9.0 - 18.0 * _r2 + 2.0 * jnp.sqrt(3.0)
                      * jnp.sqrt(jnp.clip(7.0 - 27.0 * _r2 + 27.0 * _r2 ** 2, 0.0, None)))
    cth2 = jnp.clip(1.0 / (jnp.cbrt(3.0) * _term2) - _term2 / jnp.cbrt(9.0), -1.0, 1.0)
    phi2 = 2 * jnp.pi * _ev_fold_uniform(sk, 106)
    # Channel charges computed BEFORE the splits so the NN->N Delta-> N N pi products carry PHYSICAL
    # per-species/charge masses (ACHILLES decays the Delta via DecayHandler -> ParticleInfo masses).
    # Fold keys 107/108 are order-independent, so this move is bit-neutral vs the draw sequence.
    q_pair = isp.astype(jnp.int32) + nisp[ar, j].astype(jnp.int32)
    u107 = _ev_fold_uniform(sk, 107)
    u108 = _ev_fold_uniform(sk, 108)
    dch = jnp.where(q_pair == 2, jnp.where(u107 < 0.75, 2, 1),
            jnp.where(q_pair == 1, jnp.where(u107 < 0.5, 1, 0),
                                   jnp.where(u107 < 0.25, 0, -1)))
    pi_q = jnp.where(dch == 2, 1,
            jnp.where(dch == 1, jnp.where(u108 < 1.0 / 3.0, 1, 0),
            jnp.where(dch == 0, jnp.where(u108 < 2.0 / 3.0, 0, -1), -1)))
    _mN1 = jnp.where((q_pair - dch) == 1, _MP_PHYS, _MN_PHYS)   # nucleon recoiling against the Delta
    _mN2 = jnp.where((dch - pi_q) == 1, _MP_PHYS, _MN_PHYS)     # nucleon from the Delta decay
    _mpi_dec = _CH_MASS[(1 - pi_q)]                             # pion from the Delta decay (per-charge phys)
    # Delta-mass window (ACHILLES ResonanceHelper.cc:25-27): floor = neutron+pi+ ("heavier" convention),
    # ceiling = sqrts - PHYSICAL recoil-nucleon mass (was avg M_N +/- ad-hoc 1 MeV buffer).
    _m_d_hi = jnp.maximum(rs_j - _mN1, _MN_PHYS + _CH_MASS[0] + 1.0)
    m_d = jnp.clip(_m_d_raw, _MN_PHYS + _CH_MASS[0], _m_d_hi)

    def _split2(P4, mA, mB, cth_, phi_, aniso_axis=False):
        ss = jnp.clip(P4[:, 0] ** 2 - jnp.sum(P4[:, 1:] ** 2, axis=1), (mA + mB) ** 2 * 1.0001, None)
        rss = jnp.sqrt(ss)
        EA = (ss + mA ** 2 - mB ** 2) / (2 * rss)
        pf = jnp.sqrt(jnp.clip(EA ** 2 - mA ** 2, 0.0, None))
        sth_ = jnp.sqrt(jnp.clip(1 - cth_ ** 2, 0, None))
        if aniso_axis:
            # D6: measure the polar angle cth_ relative to the DECAYING PARTICLE's momentum direction,
            # not the lab z-axis.  ACHILLES rotates the rest-frame products so z aligns with the mother's
            # momentum before boosting (DecayHandler.cc:76-79; the NN->NDelta Delta has no Mothers set,
            # so the `else` branch = the Delta's OWN momentum is used).  For an ISOTROPIC decay the axis
            # is irrelevant, but the AngularMom:2 law (1+3cos^2) is anisotropic, so the axis matters:
            # build the rest-frame unit vector in the {phat, e1, e2} basis with phat = parent momentum.
            phat = P4[:, 1:] / jnp.clip(jnp.linalg.norm(P4[:, 1:], axis=1, keepdims=True), 1e-9, None)
            ref = jnp.where(jnp.abs(phat[:, 0:1]) < 0.9,
                            jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
            e1 = ref - jnp.sum(ref * phat, axis=1, keepdims=True) * phat
            e1 = e1 / jnp.clip(jnp.linalg.norm(e1, axis=1, keepdims=True), 1e-9, None)
            e2 = jnp.cross(phat, e1)
            d_ = (cth_[:, None] * phat
                  + sth_[:, None] * (jnp.cos(phi_)[:, None] * e1 + jnp.sin(phi_)[:, None] * e2))
        else:
            d_ = jnp.stack([sth_ * jnp.cos(phi_), sth_ * jnp.sin(phi_), cth_], axis=1)
        pa = jnp.concatenate([EA[:, None], pf[:, None] * d_], axis=1)
        pb = jnp.concatenate([(rss - EA)[:, None], -pf[:, None] * d_], axis=1)
        beta_ = P4[:, 1:] / P4[:, [0]]
        b2_ = jnp.sum(beta_ ** 2, axis=1); g_ = 1 / jnp.sqrt(jnp.clip(1 - b2_, 1e-12, None))
        def lab(p4_):
            bp_ = jnp.sum(beta_ * p4_[:, 1:], axis=1)
            E = g_ * (p4_[:, 0] + bp_)
            p3 = p4_[:, 1:] + ((g_ - 1) * bp_ / jnp.clip(b2_, 1e-30, None) + g_ * p4_[:, 0])[:, None] * beta_
            return jnp.concatenate([E[:, None], p3], axis=1)
        return lab(pa), lab(pb)

    pN1, pD = _split2(Pj, _mN1, m_d, cth1, phi1)                         # production: isotropic
    pN2, _pPiX = _split2(pD, _mN2, _mpi_dec, cth2, phi2, aniso_axis=True)  # D6: (1+3cos^2) about pD dir
    # Pauli-block the two inelastic outgoing nucleons per species; the LEADING one (faster -> continues
    # from |pos|) is blocked at the LEADING's position (like the elastic kf_lead), the other (knockout at
    # the struck vertex) at the struck k_F -- same position fix as the elastic channel.
    _kfp_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoN))
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)   # pN1 is leading
    _n1p = (q_pair - dch) == 1; _n2p = (dch - pi_q) == 1
    kf_N1 = jnp.where(nl_is1, jnp.where(_n1p, _kfp_lead, _kfn_lead), jnp.where(_n1p, kf_p_j, kf_n_j))
    kf_N2 = jnp.where(~nl_is1, jnp.where(_n2p, _kfp_lead, _kfn_lead), jnp.where(_n2p, kf_p_j, kf_n_j))
    in_blocked = ((jnp.linalg.norm(pN1[:, 1:], axis=1) < kf_N1)
                  | (jnp.linalg.norm(pN2[:, 1:], axis=1) < kf_N2))
    if not cfg.pauli:
        in_blocked = in_blocked & False
    is_inel = chose_inel & ~in_blocked
    lead_in = jnp.where(nl_is1[:, None], pN1, pN2)
    pi_chidx = (1 - pi_q).astype(jnp.int32)
    do = has_hit & ~chose_inel & ~blocked
    recoil = (p4 + pN_j) - p_out
    bg_proton = rec_isp_out                                      # elastic recoil isospin (charge-exchange aware)
    fz_new = _formation_zone(p4, p_out)
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)
    inel_nl = jnp.where(nl_is1[:, None], pN2, pN1)
    inel_nl_q = jnp.where(nl_is1, dch - pi_q, q_pair - dch)
    ko_cand = jnp.where(is_inel[:, None], inel_nl, recoil)
    ko_q = jnp.where(is_inel, inel_nl_q, bg_proton.astype(jnp.int32))
    # CONTINUING (leading) nucleon's new charge: elastic charge-exchange -> lead_isp_out; inelastic -> the
    # LEADING nucleon's channel charge (the partner of the knockout inel_nl_q); else unchanged.  Previously
    # the leading kept its incident charge always (no NN charge exchange + inelastic leading mis-charged).
    lead_inel_q = jnp.where(nl_is1, q_pair - dch, dch - pi_q)
    lead_q_new = jnp.where(do, lead_isp_out.astype(jnp.int32),
                           jnp.where(is_inel, lead_inel_q, isp.astype(jnp.int32))).astype(jnp.int32)
    ko_fz = jnp.where(is_inel, _formation_zone(p4, inel_nl), _formation_zone(p4, recoil))
    ko_alive = (do | is_inel) & (jnp.linalg.norm(ko_cand[:, 1:], axis=1) > 1.0)
    ko_pos = npos[ar, j]
    # created pion spawn (inelastic only)
    # D8: Delta+ (2214) and Delta0 (2114) have a radiative branch Delta->N gamma at BR 0.0055
    # (data/decays.yml); Delta++ and Delta- go 100% to N pi.  The Delta charge is `dch`, so this branch
    # is open only for dch in {0,+1}.  When it fires the vertex emits a photon instead of a pion -> no
    # pion produced (the recoil nucleon N2 is kept with its N pi kinematics; the ~massless-vs-m_pi
    # kinematic shift on a 0.55% branch is negligible, and the photon does not reinteract).
    _delta_pm = (dch == 0) | (dch == 1)
    _gamma = _delta_pm & (_ev_fold_uniform(sk, 111) < 0.0055)
    pi_alive = is_inel & (jnp.linalg.norm(_pPiX[:, 1:], axis=1) > 1.0) & ~_gamma
    pi_fz = _formation_zone(p4, _pPiX)
    # BIRTH POSITION of the Delta's decay products (ACHILLES NucleonNucleon.cc, ResonanceMode::Decay):
    # when the Delta is out_ids[0] ("slot a") the products are inserted at particle1.Position() -- the
    # LEADING nucleon -- because the `decay.Position() = particle2.Position()` override there runs AFTER
    # decays_out.insert() and is dead code; when the Delta is out_ids[1] ("slot b") they get
    # particle2.Position(), the STRUCK nucleon.  a/b is the same 50/50 mode split as the elastic branch.
    # ADoNIS always used the struck vertex -> the pion started up to ~an impact parameter deeper/shallower
    # than ACHILLES for half the vertices, changing how much material it traverses before escaping.
    # VERIFIED against NucleonNucleon.cc: allowed_states (:38-45) lists the Delta FIRST in EVERY
    # NN->NDelta channel, so out_ids[0] is always the resonance -> info_a.IsResonance() is always true
    # and info_b never is.  Only one branch ever runs: decays_a is built at particle1.Position() (:161)
    # and inserted (:162) BEFORE the `decay.Position() = particle2.Position()` override (:163), which is
    # therefore dead code operating on an already-copied vector.  Net: the Delta's products -- the
    # nucleon AND the pion -- are ALWAYS born at particle1.Position(), the LEADING nucleon; only the
    # slot-b nucleon (:172) sits at particle2.Position().  There is no 50/50 slot coin.
    # ADoNIS's leading continues at `pos` and the knockout spawns at `npos[ar,j]`, matching :161/:172,
    # so the pion belongs at `pos`.  (Was: always npos -> struck vertex; then a 50/50 mix -> half right.)
    pi_pos = jnp.broadcast_to(pos, _pPiX[:, 1:].shape)
    # leading update + consumed depletion + fz + advance
    p4 = jnp.where(do[:, None], p_out, jnp.where(is_inel[:, None], lead_in, p4))
    consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & (do | is_inel)[:, None])
    fz = jnp.where((fz > 0.0) & alive, fz - timeStep, fz)
    fz = jnp.where(do, fz_new, jnp.where(is_inel, _formation_zone(p4, lead_in), fz))
    # D3: advance along the PRE-interaction direction (the incoming `dhat`).  ACHILLES sets the position
    # inside AllowedInteractions via Propagate (Cascade.cc:705), which runs BEFORE FinalizeMomentum, so
    # the full step is taken along the OLD momentum direction; the post-scatter direction only takes
    # effect on the NEXT step.  `dhat` is the pre-interaction unit direction (== the direction of the
    # incoming p4, used for the candidate geometry throughout this body); on a non-interacting step it
    # already equals the recomputed one, so this is a no-op there.
    pos = pos + _dstep[:, None] * dhat * alive[:, None]    # beta*step (time-sync) or step (distance-sync)
    d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)  # NEXT step
    return ((p4, pos, dhat, fz, alive, lead_q_new), escaping, captured, do.astype(jnp.int32),
            (ko_cand, ko_pos, ko_fz, ko_q, ko_alive),
            (_pPiX, pi_pos, pi_fz, pi_chidx, pi_alive), consumed,
            jax.lax.stop_gradient((has_hit, perp2_c, sig_c, iso_c, finel_c, is_inel, do & _swap_cx)))


def _pion_step(p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed,
               rgrid, rhoP, rhoN, radius, cfg, key, dt_evt=None, is_beam=None):
    """ONE step of the PION cascade for one pion per event (n,) -- a line-for-line extraction of
    `_propagate_discrete.body` (algo="step"), re-expressed for the POOLED engine: the absorption
    products (piNN->NN, up to 2 protons), the scatter recoil, and the eta-N' conversion baryon are
    returned as IMMEDIATE NUCLEON spawns (no best_abs/best_rec top-K deferral); the scattered pion
    continues in place (charge oscillates) and the absorbed/converted pion is removed.  RNG usage
    matches body exactly (split(sk,7) + fold_in(ka,211/212) + per-event splits), so iterating this with
    the same per-step keys reproduces the bfs leading PION trajectory (p_pi, ch, nsc, absorbed, conv,
    pos) bit-for-bit.  ch = pion charge INDEX (0=pi+,1=pi0,2=pi-).
    Returns: (p4', pos', dhat', ch', nsc', alive'), escaping, is_abs, is_conv,
             (s1_p4,s1_pos,s1_fz,s1_q,s1_al), (s2_p4,s2_pos,s2_fz,s2_q,s2_al), consumed', srec."""
    assert cfg.algo == "step", "pool pion step implements the 'step' algo (production path) only"
    n, A = nisp.shape; ar = jnp.arange(n)
    p_pi = p4
    is_eta = (ch == 3)                     # meson-track species: pion charge 0/1/2, or 3 = eta (propagated
    #                                        after piN->etaN conversion so it can back-convert etaN->piN)
    # ----- escape (Cascade.cc:532-553): beam pion (nsc==0) -> z>=radius PLANE; scattered -> sphere -----
    # The POOL has no early-exit "inert" skip (unlike _propagate_discrete's while_loop), so a pion that
    # has LEFT the nucleus (|pos|>radius, moving outward -> rho=0 ahead, can never re-enter) must be
    # escaped explicitly here for ALL nsc; otherwise an un-scattered (nsc==0) pion leaving in any
    # direction but +z never triggers esc_plane/esc_sphere and idles to max_steps (BFS treats these as
    # inert-survived).  inert subsumes esc_sphere; esc_plane keeps the +z beam-transparency convention.
    # D2: use the EXPLICIT external_test flag, not the `nsc==0` proxy.  Only ACHILLES's
    # ParticleStatus::external_test beam gets the z-plane rule (Cascade.cc:628-651); every secondary is
    # pushed as Status::propagating (Cascade.cc:965-973).  A CREATED pion is also nsc==0 at birth, so the
    # old proxy wrongly handed it the beam's plane escape.  Fall back to the proxy only when the caller
    # supplies no flag (legacy/non-pool callers), where the pion beam IS the nsc==0 particle.
    ext = (nsc == 0) if is_beam is None else is_beam
    # D1: pure position test, no `& outward` gate -- matches Cascade.cc:640.  See _nucleon_step.
    esc_sphere = jnp.linalg.norm(pos, axis=1) > radius
    # Cascade.cc:632 is an if/else, NOT a union: the external_test beam is tested ONLY against the
    # z-plane, everything else ONLY against the sphere.  This matters because the beam is launched at
    # z0 = -1.05*radius, i.e. ALREADY OUTSIDE the sphere -- OR-ing the two tests (as the old
    # `esc_plane | inert` did) escapes the beam at step 0 the moment the `& outward` guard is removed,
    # and the pion beam never interacts at all.  Mirrors _nucleon_step's jnp.where.
    escaping = jnp.where(ext, pos[:, 2] >= radius, esc_sphere)
    alive = alive & ~escaping
    # STEPPING CLOCK (see _nucleon_step): time-sync -> sweep beta*dt_evt (dt_evt=step/beta_max from the pool);
    # else fixed step.  Pions carry NO formation zone (ACHILLES skips IsPion in InFormationZone) -> slab+advance only.
    _beta_pi = jnp.linalg.norm(p_pi[:, 1:], axis=1) / jnp.clip(p_pi[:, 0], 1e-9, None)
    if cfg.time_step:
        _dt = jnp.full_like(_beta_pi, cfg.step) if dt_evt is None else dt_evt
        _dstep = _beta_pi * _dt
    else:
        _dstep = jnp.full_like(_beta_pi, cfg.step)
    rel = npos - pos[:, None, :]
    par = jnp.sum(rel * dhat[:, None, :], axis=2)
    perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
    cand = (par > 0) & (~consumed) & alive[:, None] & (par <= _dstep[:, None])
    pE = p_pi[:, 0]; pmom = jnp.linalg.norm(p_pi[:, 1:], axis=1); m_pi = _CH_MASS[ch]
    vpi = p_pi[:, 1:] / pE[:, None]
    rnuc = jnp.linalg.norm(npos, axis=2)
    kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP), _rho_species(rnuc, rgrid, rhoN)))

    def _xsec(nm, npo, nip):
        vN = nm[..., 1:] / nm[..., 0:1]
        vrel = jnp.clip(jnp.linalg.norm(vpi[:, None, :] - vN, axis=-1), 1e-3, None)
        rnpo = jnp.linalg.norm(npo, axis=-1)
        rho_t = _rho_species(rnpo, rgrid, rhoP) + _rho_species(rnpo, rgrid, rhoN)
        kf = _kf_local(jnp.where(nip, _rho_species(rnpo, rgrid, rhoP), _rho_species(rnpo, rgrid, rhoN)))
        Pp = p_pi[:, None, :] + nm
        Wl = jnp.sqrt(jnp.clip(Pp[..., 0] ** 2 - jnp.sum(Pp[..., 1:] ** 2, axis=-1), 1.0, None))
        sal = jnp.clip(ox.abs_cross_section(pE[:, None] + 0 * Wl, m_pi[:, None] + 0 * Wl,
                       pmom[:, None] + 0 * Wl, vrel, jnp.clip(kf, 1e-6, None),
                       jnp.clip(rho_t, 1e-9, None)), 0.0, None)
        K_ = Wl.shape[1]
        nuc_i = jnp.where(nip, 0, 1).astype(jnp.int32)
        ch_pi = jnp.clip(ch, 0, 2)                              # eta slots reuse pi+ table (unused for eta)
        Wf = Wl.reshape(-1); nuf = nuc_i.reshape(-1); chf = jnp.broadcast_to(ch_pi[:, None], (n, K_)).reshape(-1)
        sio = meson_baryon_xsec.jax_channel_sigmas_resolved(Wf, chf, nuf).reshape(n, K_, 3)
        ssl = jnp.clip(jnp.sum(sio, axis=-1), 0.0, None)
        sil = jnp.clip(meson_baryon_xsec.jax_conversion_sigma(Wf, chf, nuf).reshape(n, K_), 0.0, None)
        # --- ETA meson track (ch==3): no Oset absorption; elastic etaN->etaN is "scatter"; back-conversion
        # etaN->piN is "conversion" (regenerates a pion).  Blend by the per-event species (is_eta). --------
        ssl_eta = jnp.clip(meson_baryon_xsec.jax_eta_elastic_sigma(Wf).reshape(n, K_), 0.0, None)
        sil_eta = jnp.clip(jnp.sum(meson_baryon_xsec.jax_eta_backconv_sigma(Wf, nuf).reshape(n, K_, 3), -1), 0.0, None)
        ise = is_eta[:, None]
        sal = jnp.where(ise, 0.0, sal)
        ssl = jnp.where(ise, ssl_eta, ssl)
        sil = jnp.where(ise, sil_eta, sil)
        return sal, ssl, sil, sio, Wl

    if cfg.fast_xsec:
        score = jnp.where(cand, -perp2, -jnp.inf)
        _, idx = jax.lax.top_k(score, _KSLAB)
        gi = (ar[:, None], idx)
        sa_k, ss_k, si_k, sio_k, W_k = _xsec(nmom[ar[:, None], idx], npos[ar[:, None], idx], nisp[ar[:, None], idx])
        sa = jnp.zeros((n, A)).at[gi].set(sa_k); ss = jnp.zeros((n, A)).at[gi].set(ss_k)
        si = jnp.zeros((n, A)).at[gi].set(si_k); sig_io = jnp.zeros((n, A, 3)).at[gi].set(sio_k).reshape(n * A, 3)
        W = jnp.zeros((n, A)).at[gi].set(W_k)
    else:
        sa, ss, si, sio, W = _xsec(nmom, npos, nisp); sig_io = sio.reshape(n * A, 3)
    like_charge = ((ch == 0)[:, None] & nisp) | ((ch == 2)[:, None] & (~nisp))
    sa = sa * jnp.where(like_charge, 5.0 / 6.0, 1.0)
    sig = sa + ss + si
    _sfm = jnp.clip(sig * MB_TO_FM2, 1e-12, None)
    # Gaussian interaction probability (the single, differentiable model)
    prob = jnp.where(cand, jnp.exp(-jnp.pi * perp2 / _sfm), 0.0)
    _ks7 = _ev_split(key, 7)            # per-event keys (n,2) each
    sk, ku, kc, kf, ka, kab, knp = (_ks7[:, i] for i in range(7))
    passes = cand & (_ev_uniform(ku, (A,)) < prob)
    metric = jnp.where(passes, perp2, jnp.inf)
    j = jnp.argmin(metric, axis=1)
    has_hit = jnp.isfinite(metric[ar, j]) & alive
    sa_j = sa[ar, j]; si_j = si[ar, j]; sig_j = sig[ar, j]
    # CLOSEST in-slab candidate (mirrors _nucleon_step's perp2_c/sig_c).  The pion's INTERACTION
    # PROBABILITY is prob = exp(-pi*perp2/(sigma_tot*MB_TO_FM2)) -- it responds to a sigma scale, but the
    # per-hit branch record (sa_j..si_j, taken at the STRUCK candidate) cannot express that: under a common
    # rescale the branch LR is identically 1.  These are the sufficient statistics for the hit/no-hit
    # factor, recorded on EVERY in-slab step (hit or not), exactly as the nucleon record does.
    perp2_is = jnp.where(cand, perp2, jnp.inf)
    cidx = jnp.argmin(perp2_is, axis=1)
    has_slab = jnp.any(cand, axis=1)
    perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6)
    sa_c = sa[ar, cidx]; si_c = si[ar, cidx]
    # ss DIRECTLY at the candidate -- NOT sig_c - sa_c - si_c.  That subtraction is a cancellation
    # (~10 out of ~200) and, once stored as float32, can round to ss_c < ss_el_c, so clip(ss_c-ss_el_c,0)
    # truncates and g drifts off 1 at nominal -- which exp(-a/g) amplifies.  ss is already computed.
    ss_c = ss[ar, cidx]
    ss_el_c = sig_io.reshape(n, A, 3)[ar, cidx][ar, jnp.clip(ch, 0, 2)]   # elastic part (eta: ch clipped, unused)
    W_j = W[ar, j]; pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
    kf_p_j = _kf_local(_rho_species(rnuc[ar, j], rgrid, rhoP))
    kf_n_j = _kf_local(_rho_species(rnuc[ar, j], rgrid, rhoN))
    p_abs = sa_j / jnp.clip(sig_j, 1e-12, None)
    p_conv = si_j / jnp.clip(sig_j, 1e-12, None)
    u_br = _ev_uniform(kc)
    chose_abs = has_hit & (u_br < p_abs)
    chose_conv = has_hit & ~chose_abs & (u_br < p_abs + p_conv)
    # ----- absorption final state (isospin partition; per-species Pauli) -----
    struck_p = nisp[ar, j].astype(jnp.int32)
    d2 = jnp.sum((npos - npos[ar, j][:, None, :]) ** 2, axis=2)
    self_used = (jnp.arange(A)[None, :] == j[:, None]) | consumed
    d2p = jnp.where(self_used | (~nisp), jnp.inf, d2)
    d2n = jnp.where(self_used | nisp, jnp.inf, d2)
    pj_p = jnp.argmin(d2p, axis=1); has_p = jnp.isfinite(d2p[ar, pj_p])
    pj_n = jnp.argmin(d2n, axis=1); has_n = jnp.isfinite(d2n[ar, pj_n])
    idx2 = ch * 2 + struck_p
    Wabs = jnp.asarray(_ABS_W_NP)[idx2]; PARTabs = jnp.asarray(_ABS_PART_NP)[idx2]
    avail = jnp.where(PARTabs == 1, has_p[:, None], jnp.where(PARTabs == 0, has_n[:, None], False))
    Wm = jnp.where(avail, Wabs, 0.0); wtot = jnp.sum(Wm, axis=1, keepdims=True)
    Wm = Wm / jnp.clip(wtot, 1e-12, None)
    u_np = _ev_uniform(knp)
    nprot_out = jnp.clip(jnp.sum((u_np[:, None] > jnp.cumsum(Wm, axis=1)).astype(jnp.int32), axis=1), 0, 2)
    has_mode = wtot[:, 0] > 0
    part_is_p = PARTabs[ar, nprot_out] == 1
    pN_p = jnp.where(part_is_p[:, None], nmom[ar, pj_p], nmom[ar, pj_n])
    pos_hit = pos
    rA = jnp.linalg.norm(pos_hit, axis=1); rB = jnp.linalg.norm(npos[ar, j], axis=1)
    kfPA = _kf_local(_rho_species(rA, rgrid, rhoP)); kfNA = _kf_local(_rho_species(rA, rgrid, rhoN))
    kfPB = _kf_local(_rho_species(rB, rgrid, rhoP)); kfNB = _kf_local(_rho_species(rB, rgrid, rhoN))

    def abs_one(p_pi_i, pNj_i, pNp_i, npr, kfpa, kfna, kfpb, kfnb, k):
        P = p_pi_i + pNj_i + pNp_i
        s = P[0] ** 2 - jnp.sum(P[1:] ** 2)
        k1, k2, k3 = jax.random.split(k, 3)
        a_is_p = jax.random.uniform(k3) < 0.5
        A_is_p = (npr >= 2) | ((npr == 1) & a_is_p)
        B_is_p = (npr >= 2) | ((npr == 1) & (~a_is_p))
        # ACHILLES PionAbsorption.cc:175-181: PHYSICAL product-nucleon masses -> ASYMMETRIC CM energy
        # split (was avg M_N for both, i.e. forced symmetric).  Blocking uses outgoing |p| vs kF.
        mA = jnp.where(A_is_p, _MP_PHYS, _MN_PHYS)
        mB = jnp.where(B_is_p, _MP_PHYS, _MN_PHYS)
        s = jnp.clip(s, (mA + mB) ** 2, None)
        sqrts = jnp.sqrt(s)
        Ea = sqrts / 2.0 * (1.0 + (mA ** 2 - mB ** 2) / s)
        pstar = jnp.sqrt(jnp.clip((s - (mA + mB) ** 2) * (s - (mA - mB) ** 2), 0.0, None)) / (2.0 * sqrts)
        cth = 2.0 * jax.random.uniform(k1) - 1.0
        sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0.0, None)); phi = 2 * jnp.pi * jax.random.uniform(k2)
        dirn = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
        beta = P[1:] / P[0]
        pa = _boost(jnp.concatenate([Ea[None], pstar * dirn]), beta)
        pb = _boost(jnp.concatenate([(sqrts - Ea)[None], -pstar * dirn]), beta)
        ma = jnp.linalg.norm(pa[1:]); mb = jnp.linalg.norm(pb[1:])
        kfA = jnp.where(A_is_p, kfpa, kfna); kfB = jnp.where(B_is_p, kfpb, kfnb)
        blocked = (ma < kfA) | (mb < kfB)
        # ACHILLES re-cascades BOTH absorption nucleons (Cascade.cc particles_out[0],[1]) regardless of
        # charge: a neutron product can knock out a proton downstream.  Emit both full 4-vecs + their
        # true charges, NOT proton-only slots (zeroing neutrons dropped secondary proton knockouts).
        return pa, pb, A_is_p.astype(jnp.int32), B_is_p.astype(jnp.int32), blocked
    abs_pa, abs_pb, abs_qa, abs_qb, abs_blocked = jax.vmap(abs_one)(p_pi, pN_j, pN_p, nprot_out,
                                                         kfPA, kfNA, kfPB, kfNB, kab)   # kab (n,2) per-event
    if not cfg.pauli:
        abs_blocked = abs_blocked & False
    is_abs = chose_abs & ~abs_blocked & has_mode
    # ----- scatter: out-pion charge, DCC angle, per-species Pauli recoil -----
    # ETA elastic (etaN->etaN): the meson stays an eta (out_ch=3), isotropic CM angle, no charge exchange
    # (recoil = struck nucleon).  PION scatter: DCC out-charge + angle as before.
    sig_io_j = sig_io.reshape(n, A, 3)[ar, j]
    probs = sig_io_j / jnp.clip(jnp.sum(sig_io_j, axis=1, keepdims=True), 1e-12, None)
    u = _ev_uniform(kf, (1,))
    out_ch = jnp.clip(jnp.sum((u > jnp.cumsum(probs, axis=1)).astype(jnp.int32), axis=1), 0, 2).astype(jnp.int32)
    out_ch = jnp.where(is_eta, jnp.int32(3), out_ch)          # eta elastic -> stays eta
    nuc_idx = jnp.where(nisp[ar, j], 0, 1)
    chan_idx = jnp.clip(ch, 0, 2) * 6 + nuc_idx * 3 + jnp.clip(out_ch, 0, 2)
    u_ang = _ev_uniform(ka)
    cos_cm = jnp.where(is_eta, 2.0 * u_ang - 1.0,             # eta: isotropic; pion: DCC angular table
                       meson_baryon_xsec.jax_sample_cos_cm(W_j, u_ang, chan_idx))
    _rec_is_p = jnp.where(is_eta, struck_p == 1, (struck_p + out_ch - ch) == 1)   # recoil nucleon charge
    kf_rec_pi = jnp.where(_rec_is_p, kf_p_j, kf_n_j)
    m_rec_pi = jnp.where(_rec_is_p, _MP_PHYS, _MN_PHYS)        # recoil nucleon mass PHYSICAL per-species

    def scat_one(p_pi_i, pN_i, out_i, kf_i, mrec, cc, k):
        p_out = _two_body_cm_scatter(p_pi_i, pN_i, _CH_MASS[out_i], k, cos_cm=cc, m_recoil=mrec)
        p_rec = (p_pi_i + pN_i) - p_out
        return p_out, jnp.linalg.norm(p_rec[1:]) < kf_i
    p_out, blocked = jax.vmap(scat_one)(p_pi, pN_j, out_ch, kf_rec_pi, m_rec_pi, cos_cm, sk)   # sk (n,2) per-event
    if not cfg.pauli:
        blocked = blocked & False
    # ----- conversion: pion<->eta MORPH (+ pion->K terminal) -----------------------------------------
    # A pion conversion (piN->etaN) now emits a propagating eta (meson spawn, charge idx 3) with prob
    # si_eta/si_total; the rest (KLambda/KSigma) stays terminal.  An eta conversion (etaN->piN) emits a
    # REGENERATED pion (charge sampled from the per-charge back-conversion sigma).  The recoil N' baryon
    # is emitted (as before) in either morph direction.  ACHILLES propagates the eta the same way, which
    # is why ~1/3 of high-|p| conversions do NOT end up absorbed (the eta back-converts to a pion).
    is_conv = chose_conv
    nuc_idx_j = nuc_idx                                        # struck-nucleon index (0 p, 1 n)
    bc_j = meson_baryon_xsec.jax_eta_backconv_sigma(W_j, nuc_idx_j)   # (n,3) etaN->pi_c N sigma per out-pion
    pbc = bc_j / jnp.clip(jnp.sum(bc_j, axis=1, keepdims=True), 1e-12, None)
    u_out = _ev_fold_uniform(ka, 331)
    out_pi_idx = jnp.clip(jnp.sum((u_out[:, None] > jnp.cumsum(pbc, axis=1)).astype(jnp.int32), axis=1),
                          0, 2).astype(jnp.int32)
    si_eta_j = meson_baryon_xsec.jax_pi_to_eta_sigma(W_j, jnp.clip(ch, 0, 2), nuc_idx_j)   # piN->etaN piece (n,)
    frac_morph = jnp.where(is_eta, 1.0, si_eta_j / jnp.clip(si_j, 1e-12, None))     # eta piece of the conv
    chose_morph = is_conv & (_ev_fold_uniform(ka, 332) < frac_morph)
    m_out = jnp.where(is_eta, _CH_MASS[out_pi_idx], _CH_MASS[3])   # outgoing meson mass (eta->pion; pi->eta)
    meson_q = jnp.where(is_eta, out_pi_idx, 3).astype(jnp.int32)
    # recoil baryon charge (eta neutral): pi->eta q_bary=(1-ch)+struck_p; eta->pi q_bary=struck_p-(1-out_pi_idx)
    q_bary = jnp.where(is_eta, struck_p - (1 - out_pi_idx), (1 - ch) + struck_p)
    morph_ok = chose_morph & ((q_bary == 0) | (q_bary == 1))
    # CM two-body (meson m_out + recoil nucleon), isotropic direction.  ACHILLES MesonBaryonInteractions.cc
    # :162-182 uses the PHYSICAL recoil-baryon mass (per q_bary), not the average M_N.
    _mB_conv = jnp.where(q_bary == 1, _MP_PHYS, _MN_PHYS)
    Pcv = p_pi + pN_j
    scv = Pcv[:, 0] ** 2 - jnp.sum(Pcv[:, 1:] ** 2, axis=1)
    rscv = jnp.sqrt(jnp.clip(scv, (_mB_conv + m_out) ** 2, None))
    EN = (scv + _mB_conv ** 2 - m_out ** 2) / (2.0 * rscv)     # recoil baryon CM energy
    Em = rscv - EN                                            # outgoing meson CM energy
    pst = jnp.sqrt(jnp.clip(EN ** 2 - _mB_conv ** 2, 0.0, None))
    ccv = 2.0 * _ev_fold_uniform(ka, 211) - 1.0
    scv_ = jnp.sqrt(jnp.clip(1 - ccv ** 2, 0.0, None)); phcv = 2 * jnp.pi * _ev_fold_uniform(ka, 212)
    dcv = jnp.stack([scv_ * jnp.cos(phcv), scv_ * jnp.sin(phcv), ccv], axis=1)
    beta = Pcv[:, 1:] / Pcv[:, [0]]; b2 = jnp.sum(beta ** 2, axis=1); gcv = 1 / jnp.sqrt(jnp.clip(1 - b2, 1e-12, None))

    def _boost_cm(Ecm, p3cm):                                 # boost a CM 4-vec (Ecm, p3cm) by beta
        bp = jnp.sum(beta * p3cm, axis=1)
        p3 = p3cm + ((gcv - 1) * bp / jnp.clip(b2, 1e-30, None) + gcv * Ecm)[:, None] * beta
        return jnp.concatenate([(gcv * (Ecm + bp))[:, None], p3], axis=1)
    p_bary = _boost_cm(EN, pst[:, None] * dcv)
    p_meson = _boost_cm(Em, -pst[:, None] * dcv)               # the propagated eta / regenerated pion
    # PAULI-BLOCK the piN<->etaN morph recoil nucleon N', exactly like the scatter (`blocked`) and
    # absorption (`abs_blocked`) recoils.  ACHILLES FinalizeMomentum applies `hit &= !PauliBlocking(part)`
    # GENERICALLY to every accepted channel's outgoing baryon (Cascade.cc:814-820); the conversion channel
    # previously skipped it, so ADoNIS accepted sub-k_F morph recoils ACHILLES rejects (over-producing
    # eta-conversion knockouts).  A blocked morph -> the conversion is rejected and the pion/eta continues
    # unchanged (not removed, not consumed), mirroring ACHILLES's `if(hit)` no-op.  (Only the MORPH recoil
    # is a nucleon; the terminal K-Lambda/K-Sigma conversions produce a hyperon and are not nucleon-Pauli
    # blocked, so conv_blocked is gated on `chose_morph`.)
    kf_bary = jnp.where(q_bary == 1, kf_p_j, kf_n_j)
    conv_blocked = chose_morph & (jnp.linalg.norm(p_bary[:, 1:], axis=1) < kf_bary)
    if not cfg.pauli:
        conv_blocked = conv_blocked & False
    morph_ok = morph_ok & ~conv_blocked                        # blocked morph: no meson/recoil spawn
    is_conv = is_conv & ~conv_blocked                          # blocked morph: pion NOT removed/consumed
    is_scat = has_hit & ~chose_abs & ~chose_conv & ~blocked
    p_rec = (p_pi + pN_j) - p_out
    q_rec = struck_p + out_ch - ch
    rcand = jnp.where(is_conv[:, None], p_bary, p_rec)
    fz_rec = _formation_zone(p_pi, rcand)
    # ----- spawns: slot1 = abs nucleon A | scatter recoil | conv baryon ; slot2 = abs nucleon B -----
    # Both absorption nucleons (A,B) carry their TRUE charge (abs_qa/abs_qb) and full 4-vec; neutron
    # products re-cascade (ACHILLES particles_out[0],[1]) instead of being dropped.
    s1_p4 = jnp.where(is_abs[:, None], abs_pa, jnp.where(is_scat[:, None], p_rec, jnp.where(morph_ok[:, None], p_bary, 0.0)))
    s1_q = jnp.where(is_abs, abs_qa, jnp.where(is_scat, q_rec, jnp.where(morph_ok, q_bary, 0))).astype(jnp.int32)
    s1_pos = jnp.where(is_abs[:, None], pos_hit, npos[ar, j])
    s1_fz = jnp.where(is_abs, _formation_zone(p_pi, abs_pa), jnp.where(is_scat | morph_ok, fz_rec, 0.0))
    s1_al = (jnp.linalg.norm(s1_p4[:, 1:], axis=1) > 1.0) & (is_abs | is_scat | morph_ok)
    s2_p4 = jnp.where(is_abs[:, None], abs_pb, 0.0)
    s2_q = jnp.where(is_abs, abs_qb, 0).astype(jnp.int32)
    s2_pos = pos_hit
    s2_fz = _formation_zone(p_pi, abs_pb)
    s2_al = (jnp.linalg.norm(s2_p4[:, 1:], axis=1) > 1.0) & is_abs
    # ----- meson spawn (eta from pi conversion, or pion regenerated by eta back-conversion) -----
    sm_p4 = jnp.where(morph_ok[:, None], p_meson, 0.0)
    sm_q = jnp.where(morph_ok, meson_q, 0).astype(jnp.int32)
    sm_pos = npos[ar, j]
    sm_fz = jnp.zeros_like(s1_fz)                             # mesons carry no formation zone (ACHILLES)
    sm_al = morph_ok & (jnp.linalg.norm(sm_p4[:, 1:], axis=1) > 1.0)
    # ----- pion state update (scatter continues; abs/conv removed) -----
    p_pi = jnp.where(is_scat[:, None], p_out, p_pi)
    ch = jnp.where(is_scat, out_ch, ch)
    nsc = nsc + is_scat.astype(jnp.int32)
    alive = alive & ~is_abs & ~is_conv
    interacted = is_abs | is_scat | is_conv
    consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & interacted[:, None])
    # D3 (see _nucleon_step): advance along the PRE-scatter direction (incoming `dhat`), then recompute
    # for the next step.  ACHILLES Propagate (Cascade.cc:705) runs before FinalizeMomentum.
    pos = pos + _dstep[:, None] * dhat * alive[:, None]    # beta*step (time-sync) or step (distance-sync)
    d3 = p_pi[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)  # NEXT
    ss_j = ss[ar, j]                                            # direct (see ss_c: no sig-sa-si cancellation)
    ss_el_j = sig_io_j[ar, jnp.clip(ch, 0, 2)]                  # elastic (out==in) scatter sigma at the hit
    # ETA-track steps are NOT recorded in the pion kind-1 FSI reweight (their sa/ss/si carry the eta
    # sigmas, not pion ones -> a pion knob must not scale them).  Excluding via perp2_c>1e5 makes the
    # eta path forward-faithful and gradient-neutral for the pion FSI knobs.
    perp2_c = jnp.where(is_eta, 1e6, perp2_c)
    # GRANULAR kind-1 channel code {0 elastic, 1 charge-exchange, 2 absorption, 3 conversion} (was the
    # 3-code {0 scatter,1 abs,2 conv}).  Scatter splits into elastic/cex by the sampled out-pion charge
    # (out_ch==ch -> elastic).  ss_el recorded alongside ss (total scatter) -> ss_cex = ss - ss_el; this
    # lets fsi_pion_reweight scale s_piN_elastic / s_piN_cex / s_pi_abs / s_conv independently.
    code4 = jnp.where(chose_abs, 2, jnp.where(chose_conv, 3,
                      jnp.where(out_ch == ch, 0, 1))).astype(jnp.int32)
    return ((p_pi, pos, dhat, ch, nsc, alive), escaping, is_abs, is_conv,
            (s1_p4, s1_pos, s1_fz, s1_q, s1_al), (s2_p4, s2_pos, s2_fz, s2_q, s2_al),
            (sm_p4, sm_pos, sm_fz, sm_q, sm_al), consumed,
            jax.lax.stop_gradient((has_hit, code4, sa_j, ss_el_j, ss_j, si_j,
                                   perp2_c, sa_c, ss_el_c, ss_c, si_c)))


PION, NUCLEON = 0, 1
FATE_NONE, FATE_ESCAPE, FATE_ABSORB, FATE_CONVERT, FATE_CAPTURE = 0, 1, 2, 3, 4
_ORIG_PRIM_PI = 2          # pool origin tag for the RES PRIMARY pion (0=RES recoil/QE chain, 1=pi-knockout)
_TRACK_OFFSET = 1000       # daughter track_ids start here (> any primary track_id); see make_pool_stepper
# Engine DEFAULTS (cascade_nucleus): refill + waiting-queue are ON by default so every consumer gets the
# persistent-refill engine and the keep-overflow-particles correctness (the plan's allowed change).
_DEFAULT_NW = 2048         # refill working-set width (events in flight).  n_w=None -> min(_DEFAULT_NW, n);
                           # n_w=0 forces lock-step (the bit-exact reference).  Tuned by the (workers,n_w,P) study.
# FSI reweight-record layout.  FLAT/STREAMING (opt-in): one flat (TOTAL,) buffer per field + a global
# cursor; the budget is tail-INSENSITIVE (~ n*E[interactions]) so it is NOT set by the per-event tail.
# When flat, rec_caps=(Tp,Tn) is the TOTAL interaction budget (NOT a per-event K).  DENSE (default): the
# legacy per-event (n,K) buffer.  Both give a NUMERICALLY-IDENTICAL reweight -- the reduction is a scatter-
# add by eidx (order-independent), so results agree to float64 precision (exact at nominal; 1-2 ULP off-
# nominal, from the interleaved-by-step vs contiguous-by-event log-sum order).  Default stays DENSE until
# every caller passes flat budgets; callers opt in via
# ADONIS_FLAT_FSI=1 + a total budget.  See docs/logbook/fsi_record_cap_techdebt.md.
FLAT_FSI_REC = os.environ.get("ADONIS_FLAT_FSI", "0") != "0"
_DEFAULT_QCAP = 64         # particle waiting-queue width.  q_cap=None -> this; keeps overflow particles
                           # (stack overflow sofl -> 0); q_cap=0 disables (legacy drop-on-overflow).
_HARD_STEPS = 100000       # ACHILLES cMaxSteps-style ABSOLUTE step ceiling.  Physical termination is
                           # escape/capture/absorption/path-budget; reaching this ceiling means a particle
                           # never terminated -> a RUNAWAY, which must NOT be silently truncated.  The pool
                           # RAISES (run_cascade_pool) if it is ever hit, making the anomaly visible.


def _raise_if_runaway(counter, where):
    """Guard: raise if the cascade hit the _HARD_STEPS ceiling (a particle never terminated).
    Works eager AND under jit: since generation now defaults to the jitted engine, the tracer case is
    routed through jax.debug.callback so a runaway is never SILENTLY truncated (incl. on GPU)."""
    def _check(c):
        c = int(c)
        if c > 0:
            raise RuntimeError(
                f"cascade {where}: hit the {_HARD_STEPS}-step hard ceiling for {c} particle/event(s) -- "
                "no physical termination (escape/capture/absorption/path-budget) fired.  This should never "
                "happen; investigate (runaway particle) rather than accept a silently truncated cascade.")
    try:
        _check(counter)                                              # eager: concrete scalar
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        jax.debug.callback(_check, counter)                          # jit/GPU: host-side check at run time


def setup_nucleus(p_pi, pid_pi, pid_Ni, cfg, key):
    """Sample the nucleus background + struck vertex (material from cfg) EXACTLY as DiscreteCascadeFSI.apply,
    so the engine's primary-pion segment reproduces the production chain bit-for-bit.  Returns nucleus + pion init."""
    kn, kv, kp = jax.random.split(key, 3)
    n = p_pi.shape[0]
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    A = nisp.shape[1]
    struck_isp = (pid_Ni == 2212)
    rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
    vtx = jnp.argmax(rsel, axis=1)
    pos0 = npos[jnp.arange(n), vtx]
    consumed0 = jax.nn.one_hot(vtx, A).astype(bool)
    ch0 = jnp.where(pid_pi == 211, 0, jnp.where(pid_pi == -211, 2, 1)).astype(jnp.int32)
    return dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=consumed0, ch0=ch0, kp=kp)


def empty_batch(n, P):
    """An all-dead particle buffer of shape (n, P).  origin = gen-0 ancestor (0 = RES/QE nucleon chain,
    1 = pion-scatter-knockout chain), gen = BFS depth -- PROVENANCE for offline debugging."""
    return dict(species=jnp.zeros((n, P), jnp.int32), charge=jnp.zeros((n, P), jnp.int32),
                p4=jnp.zeros((n, P, 4)), pos=jnp.zeros((n, P, 3)), fz=jnp.zeros((n, P)),
                nsc=jnp.zeros((n, P), jnp.int32),                     # pion scatter count (pool: beam-vs-internal escape)
                external_test=jnp.zeros((n, P), bool),                # ACHILLES ParticleStatus::external_test: the
                #   CrossSection beam BEFORE its first interaction -> z-plane escape (Cascade.cc:632).  Set True
                #   at beam init; cleared on the first realized interaction.  Default False = sphere (all else).
                alive=jnp.zeros((n, P), bool), w=jnp.ones((n, P)), fate=jnp.zeros((n, P), jnp.int32),
                origin=jnp.zeros((n, P), jnp.int32), gen=jnp.zeros((n, P), jnp.int32),
                track_id=jnp.zeros((n, P), jnp.int32),                # MC-truth: unique id (tracking.py)
                parent_id=jnp.full((n, P), -1, jnp.int32),            # MC-truth: spawning track (-1 = primary)
                pkey=jnp.zeros((n, P, 2), jnp.uint32),                # P-INVARIANT per-particle RNG key (lineage-derived)
                lstep=jnp.zeros((n, P), jnp.int32),                   # per-particle local step counter (RNG fold + safety cap)
                lpath=jnp.zeros((n, P)),                              # per-particle accumulated PATH length [fm] (physics cap)
                gtime=jnp.zeros((n, P), jnp.int32))                   # ABSOLUTE cascade time (= global step at P>=occ);
                                                                       # processing priority -> P-invariant claim order


def _take(b, idx):
    """Gather along the particle axis (n, P) -> (n, K) with idx (n, K)."""
    n = idx.shape[0]; ar = jnp.arange(n)[:, None]
    return dict(species=b["species"][ar, idx], charge=b["charge"][ar, idx], p4=b["p4"][ar, idx],
                pos=b["pos"][ar, idx], fz=b["fz"][ar, idx], alive=b["alive"][ar, idx],
                w=b["w"][ar, idx], fate=b["fate"][ar, idx])


def compact(b, P_out, sort_priority=False):
    """Compact live particles of a (n, M) batch to the front of a (n, P_out) buffer.  Returns
    (compacted_batch, n_overflow); overflow = live particles that did not fit in P_out (dropped, never
    silently).  sort_priority=False -> position order (cumsum).  sort_priority=True -> pack the ACTIVE
    cascade stack in ACHILLES processing order: gtime cohort, then ascending CREATION index (track_id)."""
    alive = b["alive"]; n, M = alive.shape
    if not sort_priority:
        # stable rank of each live particle within its event (0-based); dead get a large rank
        rank = jnp.cumsum(alive.astype(jnp.int32), axis=1) - 1
    else:
        # lexsort keys (LAST = primary): alive-first, then ABSOLUTE TIME gtime asc, then CREATION index
        # track_id asc.  gtime (not per-particle lstep) is the cascade-synchronized cohort so daughters act
        # with their cohort, not ahead of it; WITHIN a cohort, ascending track_id reproduces ACHILLES's
        # within-timestep order (Cascade.cc:353 iterates kickedIdxs = std::set<size_t> ascending particle
        # index = creation order).  Integer keys -> no float-precision dependence; track_id is unique per
        # alive particle (primaries 0/1; daughters _TRACK_OFFSET + step*3 + chan) -> no position fallback.
        order = jnp.lexsort((b["track_id"], b["gtime"], (~alive).astype(jnp.int32)), axis=-1)
        rank = jnp.argsort(order, axis=-1)
    rank = jnp.where(alive, rank, M + P_out)                      # dead -> out of range
    n_overflow = jnp.sum((alive & (rank >= P_out)).astype(jnp.int32))
    out = empty_batch(n, P_out)
    ar = jnp.broadcast_to(jnp.arange(n)[:, None], (n, M))
    dst = jnp.where(rank < P_out, rank, P_out)                    # P_out = scratch drop slot (will be sliced off)
    for k in ("species", "charge", "fate", "origin", "gen", "track_id", "parent_id", "nsc", "lstep", "gtime",
              "external_test"):
        if k not in b:                                            # optional (e.g. BFS kernel spawns omit nsc)
            continue
        out[k] = jnp.zeros((n, P_out + 1), b[k].dtype).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    for k in ("fz", "w", "lpath"):
        out[k] = jnp.zeros((n, P_out + 1)).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    out["alive"] = jnp.zeros((n, P_out + 1), bool).at[ar, dst].set(b["alive"], mode="drop")[:, :P_out]
    out["p4"] = jnp.zeros((n, P_out + 1, 4)).at[ar, dst].set(b["p4"], mode="drop")[:, :P_out]
    out["pos"] = jnp.zeros((n, P_out + 1, 3)).at[ar, dst].set(b["pos"], mode="drop")[:, :P_out]
    if "pkey" in b:                                              # P-invariant per-particle RNG key (n,P,2)
        out["pkey"] = jnp.zeros((n, P_out + 1, 2), b["pkey"].dtype).at[ar, dst].set(b["pkey"], mode="drop")[:, :P_out]
    return out, n_overflow


def make_pool_stepper(su, cfg, with_rec=False, with_seg=False):
    """Build the pooled-engine physics stepper (S2b): advance every slot of the (n, M) stack ONE step,
    dispatched by species (NUCLEON -> _nucleon_step, PION -> _pion_step), with the consumed mask threaded
    SLOT-SERIALLY (slot m+1 sees m's depletion).  Both per-step bodies run on every slot and are selected
    by species (the v1 2x-eval tradeoff; the dead-slot waste, the dominant 10-24x factor, is gone).  Each
    slot emits up to 2 NUCLEON spawns + 1 PION spawn: a nucleon slot -> (1 knockout N, 1 NN-created pion);
    a pion slot -> (up to 2 absorption/recoil N, 0 pion).  Returns stepper(stack, key, consumed) ->
    (stack2, terminal (n,M) bool [escaped final-state particles], spawn ParticleBatch (n,3M), consumed)."""
    npos0, nmom0, nisp0 = su["npos"], su["nmom"], su["nisp"]   # default background; per-call `bg` overrides it
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    _dead = lambda n: (jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n, bool))

    def stepper(stack, key, consumed, step=0, bg=None, dt_evt=None):
        # bg=(npos,nmom,nisp) per-call -> lets run_cascade_pool swap the background when an event slot is
        # refilled (persistent-refill engine); bg=None uses the closed-over default (legacy callers).
        # dt_evt: per-event (n,) time step = step/beta_max for the ACHILLES-faithful time-sync clock
        # (cfg.time_step); None -> the steppers fall back to a fixed Dt=step.  Ignored for distance-sync.
        npos, nmom, nisp = (npos0, nmom0, nisp0) if bg is None else bg
        n, M = stack["alive"].shape
        _dt_e = dt_evt if dt_evt is not None else jnp.full(n, cfg.step)   # (n,) per-event Dt
        # `key` is the PER-EVENT step key (n,2) = fold_in(fold_in(base, evt_id), nstep) built by
        # run_cascade_pool; fold by slot index m -> a unique per-(event,step,slot) key.  This makes a
        # refilled event draw identical randoms regardless of which slot/global-step runs it.

        def slot(consumed, m):
            p4 = stack["p4"][:, m]; pos = stack["pos"][:, m]; fz = stack["fz"][:, m]
            nsc = stack["nsc"][:, m]; chg = stack["charge"][:, m]; al = stack["alive"][:, m]
            sp = stack["species"][:, m]
            is_N = (sp == NUCLEON) & al; is_pi = (sp == PION) & al
            d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
            # P-INVARIANT RNG: key by the particle's OWN lineage key (pkey) + its OWN local step count
            # (lstep), NOT by slot index m or the global nstep.  A particle thus draws identical randoms
            # whether it is processed in-stack or after the wait queue, at any stack width P.
            pkey_m = stack["pkey"][:, m]; lstep_m = stack["lstep"][:, m]
            step_key = jax.vmap(lambda pk, ls: jax.random.fold_in(pk, ls))(pkey_m, lstep_m)   # (n,2)
            _kk = jax.vmap(lambda k: jax.random.split(k))(step_key)        # (n,2,2)
            kN, kP = _kk[:, 0], _kk[:, 1]                                  # per-particle nucleon/pion keys (n,2)
            # NUCLEON branch (charge = isospin, 1=p); inactive slots produce no consumption/spawn.
            # escN is EMITTED-to-final-state only: _nucleon_step partitions boundary-reachers into
            # escaping (emitted) vs capturedN (bound, KE<10 MeV) at the source, so escN never contains a
            # captured nucleon and `term` below is correct without any flag-application step here.
            (p4n, posn, _dn, fzn, alnN, qln), escN, capturedN, _do, koN, pinN, consumedN, nstat = _nucleon_step(
                p4, pos, dhat, fz, chg.astype(bool), is_N, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kN, dt_evt=_dt_e,
                is_beam=stack["external_test"][:, m])   # ACHILLES external_test -> z-plane escape (Deviation 2)
            # PION branch (charge = pion index 0/1/2); scatter continues, abs/conv removed.
            (p4p, posp, _dp, chp, nscp, alnP), escP, is_abs, is_conv, s1, s2, smes, consumedP, pstat = _pion_step(
                p4, pos, dhat, chg, nsc, is_pi, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kP, dt_evt=_dt_e,
                is_beam=stack["external_test"][:, m])   # D2: real external_test flag, not the nsc==0 proxy
            # kind-1 FSI reweight sufficient statistics (mirrors the legacy brec/srec per-step records):
            #   pion: record every geometric hit (has_hit) -> branch code + sigma components (sa,ss,si).
            #   nucleon: record every in-slab candidate step (perp2_c<1e5) -> hit flag + a_nom=pi b^2/sigma.
            # Computed ONLY when with_rec (the differentiable/tuning path) -> forward generation pays nothing.
            if with_rec:
                (p_hh, p_bc, p_sa, p_ss_el, p_ss, p_si,               # branch stats @ the STRUCK candidate
                 p_perp2_c, p_sa_c, p_ss_el_c, p_ss_c, p_si_c) = pstat   # survival stats @ the CLOSEST one
                n_hh, n_perp2, n_sig, n_iso, n_finel, n_inel, n_swap = nstat  # _nucleon_step granular stats
                a_nom = jnp.pi * n_perp2 / jnp.clip(n_sig * MB_TO_FM2, 1e-12, None)
                p_sig_c = jnp.clip(p_sa_c + p_ss_c + p_si_c, 1e-12, None)
                p_a_nom = jnp.pi * p_perp2_c / jnp.clip(p_sig_c * MB_TO_FM2, 1e-12, None)
                # PION: record every IN-SLAB CANDIDATE step (perp2_c < 1e5), not just the hits -- the no-hit
                # steps are exactly where the sigma_tot (mean-free-path) response lives.  Mirrors the nucleon.
                # NON-FINITE sigma is EXCLUDED, and that is not a patch: the Oset absorption sigma is NaN
                # whenever the pion has E < m_pi (ReducedHalfWidth re-derives |p| as sqrt(E^2-m^2), and the
                # RES pion is built with the mpi0 KINEMATIC mass while the cascade uses the per-charge
                # PHYSICAL mass -> negative under the root for |p| < ~35 MeV/c).  The WALK then cannot
                # interact there (prob = exp(-pi b^2/NaN) = NaN, and `u < NaN` is False), and it cannot
                # interact at ANY theta either -- scaling NaN is still NaN.  Such a candidate carries zero
                # theta-dependence, so recording it would inject NaN into the weight for no physics.
                # (ACHILLES has the identical re-derivation + PID mass, so the FORWARD behaviour is
                # faithful; the soft-pion sigma itself is a separate open question -- see logbook.)
                p_fin = jnp.isfinite(p_a_nom) & jnp.isfinite(p_sa_c) & jnp.isfinite(p_ss_c) \
                    & jnp.isfinite(p_si_c) & jnp.isfinite(p_ss_el_c)
                rec_slot = (is_pi & (p_perp2_c < 1e5) & p_fin, p_bc, p_sa, p_ss_el, p_ss, p_si,
                            p_hh, p_a_nom, p_sa_c, p_ss_el_c, p_ss_c, p_si_c,
                            is_N & (n_perp2 < 1e5), n_hh, a_nom, n_iso, n_finel, n_inel, n_swap)  # nucleon
            consumed = consumedN | consumedP                          # only the active species adds bits
            p4_2 = jnp.where(is_N[:, None], p4n, jnp.where(is_pi[:, None], p4p, p4))
            pos_2 = jnp.where(is_N[:, None], posn, jnp.where(is_pi[:, None], posp, pos))
            fz_2 = jnp.where(is_N, fzn, fz)                           # only the nucleon updates fz
            # nsc = POST-PAULI interaction count of this particle.  The pion path is unchanged (nscp).
            # The NUCLEON now counts too: _nucleon_step already computes the realized (post-Pauli)
            # elastic `_do` and inelastic `nstat[5]`, but the pool discarded them, so a nucleon's nsc
            # was always 0.  A nucleon-beam CROSS-SECTION run needs exactly this: ACHILLES counts a
            # reaction only inside `if(hit)` AFTER Pauli blocking (Cascade.cc:903), so the kind-1 `hh`
            # flag (pre-Pauli) would OVER-count.  Readers of nsc take it from the primary PION terminal
            # (pterm["nsc"]), which is untouched.
            n_react = is_N & (_do.astype(bool) | nstat[5])            # realized elastic OR inelastic
            nsc_2 = jnp.where(is_pi, nscp, nsc + n_react.astype(jnp.int32))
            # external_test (ACHILLES status) is cleared on the FIRST realized interaction of this particle
            # -- nucleon scatter/inelastic (n_react) OR pion scatter/absorb/convert -> becomes 'propagating'.
            _react = n_react | (is_pi & ((nscp > nsc) | is_abs | is_conv))
            et_2 = stack["external_test"][:, m] & ~_react
            # nucleon charge can now change: NN elastic charge-exchange / inelastic leading channel charge
            # (qln from _nucleon_step; == incident charge when no scatter).  Pion charge oscillates (chp).
            chg_2 = jnp.where(is_N, qln, jnp.where(is_pi, chp, chg))
            al_2 = jnp.where(is_N, alnN, jnp.where(is_pi, alnP, al))
            # PER-PARTICLE termination (P-INVARIANT; both caps key off the particle's OWN state, never the
            # global step count -> no P-dependent truncation; capped = dropped, not a final state):
            #   (1) PHYSICS: accumulated path length lpath >= path_budget_R * radius.  _dstep is the
            #       distance swept THIS step = beta*step (time-sync) or step (distance-sync), from the
            #       PRE-step momentum (matches the slab/advance inside _nucleon_step/_pion_step).
            #   (2) SAFETY: lstep >= max_steps -- a guaranteed-fire backstop (lstep always +1's) for the
            #       degenerate beta~0 case where lpath cannot advance.
            _beta_m = jnp.linalg.norm(d3, axis=1) / jnp.clip(p4[:, 0], 1e-9, None)
            _dstep_m = _beta_m * _dt_e if cfg.time_step else jnp.full_like(_beta_m, cfg.step)
            _lpath_2 = stack["lpath"][:, m] + _dstep_m * al.astype(stack["lpath"].dtype)
            al_2 = al_2 & (_lpath_2 < cfg.path_budget_R * radius) & ((lstep_m + 1) < cfg.max_steps)
            term = (is_N & escN) | (is_pi & escP)                    # escaped = final-state (collected)
            # per-slot fate (for the primary-pion latch in run_cascade_pool; output stays escape-only):
            # nucleon/pion escape -> ESCAPE, pion absorbed -> ABSORB, pion converted -> CONVERT.
            # D12-followup: a CAPTURED nucleon (KE<10 MeV at the boundary) is bound, NOT emitted to the
            # final state (escN excludes it), but it DID react -- record FATE_CAPTURE so the primary-fate
            # latch preserves its reaction (its nsc is otherwise lost when it leaves the output).  This is
            # ACHILLES status-26: recorded (reaction counted) but not status-1 (not a final-state particle).
            fate_2 = jnp.where(is_N & capturedN, FATE_CAPTURE,
                     jnp.where((is_N & escN) | (is_pi & escP), FATE_ESCAPE,
                     jnp.where(is_pi & is_abs, FATE_ABSORB,
                     jnp.where(is_pi & is_conv, FATE_CONVERT, FATE_NONE)))).astype(jnp.int32)
            new = (p4_2, pos_2, fz_2, nsc_2, chg_2, al_2, term, fate_2, et_2)
            # nucleon-slot knockout / pion-slot 1st product -> nuc1; pion-slot 2nd product -> nuc2.
            nuc1 = tuple(jnp.where(is_N, kn, jnp.where(is_pi, s1k, dk)) if kn.ndim == 1
                         else jnp.where(is_N[:, None], kn, jnp.where(is_pi[:, None], s1k, dk))
                         for kn, s1k, dk in zip(koN, s1, _dead(n)))
            nuc2 = tuple(jnp.where(is_pi, s2k, dk) if s2k.ndim == 1
                         else jnp.where(is_pi[:, None], s2k, dk)
                         for s2k, dk in zip(s2, _dead(n)))
            # pion daughter meson slot: nucleon inelastic pion (pinN) for a nucleon slot; for a PION slot
            # it carries the eta (piN->etaN conversion) or the regenerated pion (etaN->piN back-conversion)
            # emitted by _pion_step -- so the eta propagates & can back-convert (ACHILLES-faithful).
            pio = tuple(jnp.where(is_N, pk, jnp.where(is_pi, mk, dk)) if pk.ndim == 1
                        else jnp.where(is_N[:, None], pk, jnp.where(is_pi[:, None], mk, dk))
                        for pk, mk, dk in zip(pinN, smes, _dead(n)))
            seg_slot = None
            if with_seg:
                # per-slot SEGMENT record: channel + segment-start |p| + product (pid,|p|) set, mirroring
                # the ACHILLES VERTEXDUMP.  channel: pion {1 elastic,2 charge-ex,3 abs,4 conv}, nucleon
                # {1 elastic NN->NN, 2 inelastic NN->NNpi}, 0 = no interaction this step.  Assembled (per
                # channel) by the Python runner; here we export the raw pieces.
                _PIDPI = jnp.array([211, 111, -211])
                pi_scat = is_pi & (nscp > nsc)
                made_pi = is_N & pinN[4]                          # nucleon inelastic (created a pion)
                chan = jnp.where(made_pi, 2,
                        jnp.where(is_N & (_do > 0), 1,
                        jnp.where(is_pi & is_abs, 3,
                        jnp.where(is_pi & is_conv, 4,
                        jnp.where(pi_scat & (chp == chg), 1,
                        jnp.where(pi_scat & (chp != chg), 2, 0)))))).astype(jnp.int32)
                incp = jnp.linalg.norm(p4[:, 1:], axis=1)         # segment-start |p| (constant in flight)
                inc_pid = jnp.where(sp == NUCLEON, jnp.where(chg == 1, 2212, 2112),  # PRE-step pid
                                    _PIDPI[jnp.clip(chg, 0, 2)])
                cont_pid = jnp.where(sp == NUCLEON, jnp.where(chg_2 == 1, 2212, 2112),
                                     _PIDPI[jnp.clip(chg_2, 0, 2)])
                n1_pid = jnp.where(nuc1[3] == 1, 2212, 2112); n1_p = jnp.linalg.norm(nuc1[0][:, 1:], axis=1)
                n2_pid = jnp.where(nuc2[3] == 1, 2212, 2112); n2_p = jnp.linalg.norm(nuc2[0][:, 1:], axis=1)
                pio_pid = _PIDPI[jnp.clip(pio[3], 0, 2)]; pio_p = jnp.linalg.norm(pio[0][:, 1:], axis=1)
                cont_p = jnp.linalg.norm(p4_2[:, 1:], axis=1)
                seg_slot = (chan, incp, inc_pid, stack["parent_id"][:, m], stack["gen"][:, m],
                            stack["track_id"][:, m], cont_pid, cont_p, n1_pid, n1_p, nuc1[4].astype(jnp.int32),
                            n2_pid, n2_p, nuc2[4].astype(jnp.int32), pio_pid, pio_p, pio[4].astype(jnp.int32))
            out_slot = (new, nuc1, nuc2, pio) + ((rec_slot,) if with_rec else ()) + ((seg_slot,) if with_seg else ())
            return consumed, out_slot

        consumed, scanned = jax.lax.scan(slot, consumed, jnp.arange(M))
        new, nuc1, nuc2, pio = scanned[:4]
        T = lambda x: jnp.moveaxis(x, 0, 1)                          # (M, n, ...) -> (n, M, ...)
        p4s, poss, fzs, nscs, chgs, als, terms, fates, ets = new
        # accumulated path: +beta*step (time-sync) or +step (distance-sync) per processed step, from the
        # PRE-step momentum -- the same _dstep the slot cap uses, vectorised over the whole stack.
        _beta_all = jnp.linalg.norm(stack["p4"][:, :, 1:], axis=2) / jnp.clip(stack["p4"][:, :, 0], 1e-9, None)
        _dstep_all = _beta_all * _dt_e[:, None] if cfg.time_step else jnp.full_like(_beta_all, cfg.step)
        stack2 = {**stack, "p4": T(p4s), "pos": T(poss), "fz": T(fzs), "nsc": T(nscs),
                  "charge": T(chgs), "alive": T(als), "fate": T(fates), "external_test": T(ets),
                  "lstep": stack["lstep"] + stack["alive"].astype(jnp.int32),   # +1 per processed (alive) step
                  "lpath": stack["lpath"] + _dstep_all * stack["alive"].astype(stack["lpath"].dtype),
                  "gtime": stack["gtime"] + stack["alive"].astype(jnp.int32)}   # absolute time advances in lockstep
        terminal = T(terms)
        if with_rec:                                                 # per-slot (n,M) kind-1 record this step
            rk = ("pi_m", "pi_bc", "pi_sa", "pi_ss_el", "pi_ss", "pi_si",          # pi_m = slot mask
                  "pi_hh", "pi_a", "pi_sa_c", "pi_ss_el_c", "pi_ss_c", "pi_si_c",  # survival stats
                  "nu_m", "nu_hh", "nu_a", "nu_iso", "nu_finel", "nu_inel", "nu_swap")
            rec = {k: T(v) for k, v in zip(rk, scanned[4])}
        if with_seg:                                                 # per-slot (n,M) SEGMENT record this step
            sk = ("chan", "incp", "inc_pid", "parent_id", "gen", "track_id", "cont_pid", "cont_p",
                  "n1_pid", "n1_p", "n1_al", "n2_pid", "n2_p", "n2_al", "pio_pid", "pio_p", "pio_al")
            seg = {k: T(v) for k, v in zip(sk, scanned[4 + (1 if with_rec else 0)])}

        def _spawn(species, packs):                                  # packs: list of (p4,pos,fz,q,al) (M,n,...)
            p4_, pos_, fz_, q_, al_ = (jnp.concatenate([T(p[i]) for p in packs], axis=1) for i in range(5))
            return species, p4_, pos_, fz_, q_.astype(jnp.int32), al_
        sp_n, np4, npos_, nfz, nq, nal = _spawn(NUCLEON, [nuc1, nuc2])
        sp_p, pp4, ppos, pfz, pq, pal = _spawn(PION, [pio])
        spawn = empty_batch(n, 3 * M)
        spawn["species"] = jnp.concatenate([jnp.full((n, 2 * M), NUCLEON, jnp.int32),
                                            jnp.full((n, M), PION, jnp.int32)], axis=1)
        spawn["charge"] = jnp.concatenate([nq, pq], axis=1)
        spawn["p4"] = jnp.concatenate([np4, pp4], axis=1)
        spawn["pos"] = jnp.concatenate([npos_, ppos], axis=1)
        spawn["fz"] = jnp.concatenate([nfz, pfz], axis=1)
        spawn["alive"] = jnp.concatenate([nal, pal], axis=1)
        # P-INVARIANT daughter RNG key (PHYSICS, not just provenance): derive from the PARENT's lineage
        # key + the daughter channel + the parent's local step at spawn -> unique & order/P-independent.
        _pslot = jnp.tile(jnp.arange(M), 3)                          # (3M,) parent slot per spawn col
        _chan = jnp.repeat(jnp.arange(3), M)                         # 0=nuc1, 1=nuc2, 2=pion daughter channel
        _ppk = stack["pkey"][:, _pslot]                             # (n,3M,2) parent lineage key
        _pls = stack["lstep"][:, _pslot]                            # (n,3M) parent local step at spawn
        _dk = lambda pk, c, ls: jax.random.fold_in(jax.random.fold_in(pk, c), ls)
        spawn["pkey"] = jax.vmap(jax.vmap(_dk))(_ppk, jnp.broadcast_to(_chan, (n, 3 * M)), _pls)
        spawn["lstep"] = jnp.zeros((n, 3 * M), jnp.int32)            # daughter's own step count starts at 0
        spawn["gtime"] = stack["gtime"][:, _pslot] + 1              # but acts at the NEXT absolute time (cohort-synced)
        # track_id = CREATION index (ALWAYS set; the within-cohort processing order keys off it -- see
        # compact sort_priority).  Unique per (step,slot,channel): `step` is a SCALAR global step (lock-step)
        # or a PER-EVENT (n,) nstep (refill); both monotone in creation time, so ascending track_id == the
        # order daughters were born == ACHILLES's ascending particle index.  _pslot/_chan reuse the pkey cols.
        sa = jnp.asarray(step)
        if sa.ndim == 0:
            tid = (_TRACK_OFFSET + (step * M + _pslot) * 3 + _chan).astype(jnp.int32)
            spawn["track_id"] = jnp.broadcast_to(tid[None, :], (n, 3 * M))
        else:
            tid = (_TRACK_OFFSET + (sa[:, None] * M + _pslot[None, :]) * 3 + _chan[None, :]).astype(jnp.int32)
            spawn["track_id"] = tid
        if with_seg:                                                 # G4-like provenance (physics-inert):
            # daughter gen = parent gen + 1; parent_id = parent track_id.  origin is NOT touched (the
            # physics' RES primary-pion latch depends on its default-0 propagation).
            spawn["gen"] = stack["gen"][:, _pslot] + 1
            spawn["parent_id"] = stack["track_id"][:, _pslot]
        out = (stack2, terminal, spawn, consumed)
        if with_rec:
            out = out + (rec,)
        if with_seg:
            out = out + (seg,)
        return out

    return stepper


def pool_reconcile(stack, terminal, spawn, M, wait=None, Q=0):
    """The POOLED engine's per-step in/out (docs/logbook/cascade_pool_engine.md): drop the slots that
    terminated this step (escape/absorb/convert), KEEP the survivors, INSERT the particles created this
    step, and re-pack to the fixed width M -- counting any that don't fit as overflow.  This is exactly
    compact(concat(survivors, spawned), M), so it reuses the validated compaction primitive.
      stack    : ParticleBatch (n, M)   -- the current stack (post-step state)
      terminal : (n, M) bool            -- slots that reached a terminal this step
      spawn    : ParticleBatch (n, K)   -- particles created this step (alive mask = which are real)
      M        : int                    -- fixed active-stack width
      wait     : ParticleBatch (n, Q) or None -- the FIFO waiting buffer (particles that didn't fit in M)
      Q        : int                    -- waiting-buffer width (0 = no queue; legacy drop-on-overflow)
    With Q>0 the combined survivors+wait+spawn are packed into M active + Q waiting (drain order: survivors
    keep their slots, then the waiting buffer re-enters before brand-new spawns); only > M+Q is dropped.
    This ADDS previously-dropped particles (correctness) without changing the active ordering when Q=0.
    Returns (new_stack (n, M), new_wait (n, Q) or None, overflow (scalar))."""
    stack = {**stack, "alive": stack["alive"] & ~terminal}
    if Q > 0 and wait is not None:
        combined = {k: jnp.concatenate([stack[k], wait[k], spawn[k]], axis=1) for k in stack}
        full, ndrop = compact(combined, M + Q, sort_priority=True)  # pack to M+Q in (lstep,sid) order
        active = {k: v[:, :M] for k, v in full.items()}            # the M highest-priority -> processed next
        new_wait = {k: v[:, M:M + Q] for k, v in full.items()}     # the rest wait (re-enter by priority)
        return active, new_wait, ndrop
    combined = {k: jnp.concatenate([stack[k], spawn[k]], axis=1) for k in stack}
    active, ndrop = compact(combined, M, sort_priority=True)      # Q=0: keep the M highest-priority
    return active, None, ndrop


def _rec_scatter(bufs, cnt, mask, vals, cap):
    """Append per-event the masked slots' `vals` into persistent (n, cap) buffers at a running write
    index `cnt` (the scratch-slot `cap` is out of range -> mode='drop' discards overflow + invalid).
    Within a step the masked slots get distinct indices (cumsum order); across steps gidx>=cnt, so no
    collision with prior writes.  Returns (new_bufs, new_cnt, overflow_this_step)."""
    n, M = mask.shape; ar = jnp.arange(n)[:, None]
    order = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1
    gidx = cnt[:, None] + order
    valid = mask & (gidx < cap)
    dst = jnp.where(valid, gidx, cap)
    new_bufs = [b.at[ar, dst].set(v, mode="drop") for b, v in zip(bufs, vals)]
    overflow = jnp.sum((mask & (gidx >= cap)).astype(jnp.int32)).astype(jnp.int32)
    return new_bufs, (cnt + jnp.sum(mask.astype(jnp.int32), axis=1)).astype(cnt.dtype), overflow


def _rec_scatter_flat(bufs, gc, eidx_buf, mask, vals, evt_id, cap, n_events):
    """FLAT/STREAMING twin of _rec_scatter (docs/logbook/fsi_record_cap_techdebt.md): append this step's
    masked slots' `vals` into ONE flat (cap,) buffer at a GLOBAL running cursor `gc`, tagging each written
    slot with its event id `evt_id[row]` in `eidx_buf`.  `cap` = TOTAL interaction budget (~ n*E[interactions],
    tail-INSENSITIVE), NOT per-event K -- so it is sized by the concentrated sum, not the heavy per-event
    tail.  Overflow (gc+order >= cap) is dropped (mode='drop').  Unwritten eidx slots keep their init value
    n_events (out of range) so the by-eidx reduction (_ragged_prod) drops them.  The reweight is a scatter-ADD
    by eidx, so step-order here gives a BIT-IDENTICAL reweight to _rec_scatter's per-event order.
    Returns (new_bufs, new_gc, new_eidx_buf, overflow_this_step)."""
    n, M = mask.shape
    fm = mask.reshape(-1)                                          # (n*M,) row-major: row i's M slots contiguous
    order = jnp.cumsum(fm.astype(jnp.int32)) - 1                   # global order among THIS step's interactions
    gpos = gc + order
    valid = fm & (gpos < cap)
    dst = jnp.where(valid, gpos, cap)                             # out-of-range -> scratch idx `cap` (dropped)
    eidx_flat = jnp.repeat(evt_id, M)                             # (n*M,) event id per flattened slot
    new_bufs = [b.at[dst].set(v.reshape(-1), mode="drop") for b, v in zip(bufs, vals)]
    new_eidx = eidx_buf.at[dst].set(jnp.where(valid, eidx_flat, n_events), mode="drop")
    overflow = jnp.sum((fm & (gpos >= cap)).astype(jnp.int32)).astype(jnp.int32)
    return new_bufs, (gc + jnp.sum(fm.astype(jnp.int32))).astype(gc.dtype), new_eidx, overflow


# field lists for the flat record (mirror _P_SLOT/_N_SLOT + the reduction's expected order)
def _empty_flat_fsi_record(Tp, Tn, n_events):
    """Flat FSI record: single (Tp,)/(Tn,) field buffers + global cursors gc_p/gc_n + per-slot event index
    p_eidx/n_eidx (init n_events = out-of-range -> unwritten slots dropped by _ragged_prod).  Tp/Tn = TOTAL
    budgets (n_events * E[interactions]).  Slot defaults match _empty_fsi_record (per-slot LR=1 at nominal)."""
    return dict(bc=jnp.zeros(Tp, jnp.int32), sa=jnp.ones(Tp), ss_el=jnp.ones(Tp), ss=jnp.ones(Tp),
                si=jnp.zeros(Tp), pi_hh=jnp.zeros(Tp, bool), pi_a=jnp.full(Tp, 50.0),
                sa_c=jnp.ones(Tp), ss_el_c=jnp.ones(Tp), ss_c=jnp.ones(Tp), si_c=jnp.zeros(Tp),
                hh=jnp.zeros(Tn, bool), a=jnp.full(Tn, 50.0), iso=jnp.zeros(Tn, jnp.int32),
                finel=jnp.zeros(Tn), inel=jnp.zeros(Tn, bool), swap=jnp.zeros(Tn, bool),
                gc_p=jnp.int32(0), gc_n=jnp.int32(0),
                p_eidx=jnp.full(Tp, n_events, jnp.int32), n_eidx=jnp.full(Tn, n_events, jnp.int32))


def _empty_fsi_record(n, Kp, Kn):
    """Per-event kind-1 FSI reweight buffers (defaults give per-slot LR=1).  bc is the granular pion channel
    code {0 el,1 cex,2 abs,3 conv}; ss_el = elastic part of the total scatter sigma ss (ss_cex = ss-ss_el).
    PION slots are per IN-SLAB CANDIDATE STEP (hit or not) -- pi_hh/pi_a/*_c carry the sigma_tot (hit/no-hit)
    response, without which a common rescale of the four pion sigmas is a spurious flat direction.
    Defaults: a=50 (p0->0 => no-hit factor 1), sa_c/ss_c=1, si_c=0 => g=1 at nominal."""
    return dict(bc=jnp.zeros((n, Kp), jnp.int32), sa=jnp.ones((n, Kp)),
                ss_el=jnp.ones((n, Kp)), ss=jnp.ones((n, Kp)),
                si=jnp.zeros((n, Kp)),
                pi_hh=jnp.zeros((n, Kp), bool), pi_a=jnp.full((n, Kp), 50.0),
                sa_c=jnp.ones((n, Kp)), ss_el_c=jnp.ones((n, Kp)), ss_c=jnp.ones((n, Kp)),
                si_c=jnp.zeros((n, Kp)), nh=jnp.zeros(n, jnp.int32),
                hh=jnp.zeros((n, Kn), bool), a=jnp.full((n, Kn), 50.0),
                iso=jnp.zeros((n, Kn), jnp.int32), finel=jnp.zeros((n, Kn)),
                inel=jnp.zeros((n, Kn), bool), swap=jnp.zeros((n, Kn), bool), ns=jnp.zeros(n, jnp.int32))


def pool_fsi_reweight(record, sabs, sscat, *, s_piN_elastic=None, s_piN_cex=None, s_conv=1.0,
                      s_NN_elastic=None, s_NN_inelastic=None, f_NN_cex=0.5):
    """Joint kind-1 FSI reweight for the pool.  Pion branch-split + nucleon (per-iso el/inel).  Pure in the
    scales; == 1 at nominal; == the in-walk weight at any theta.  BACKWARD-COMPATIBLE: pool_fsi_reweight(
    record, sabs, sscat) reproduces the legacy reweight bit-for-bit (all granular knobs default to sabs/sscat).
    GRANULAR knobs (differentiable_knobs.md Group A): pion s_piN_elastic/s_piN_cex/s_conv (s_pi_abs==sabs);
    nucleon s_NN_elastic/s_NN_inelastic each a length-3 per-iso {pp,pn,nn} scale (default = sscat each)."""
    s_el = sscat if s_piN_elastic is None else s_piN_elastic
    s_cex = sscat if s_piN_cex is None else s_piN_cex
    if "p_eidx" in record:                       # RAGGED (bank) record -> same physics, ragged reduction
        return pool_fsi_reweight_flat(record, sabs, sscat, s_piN_elastic=s_piN_elastic,
                                      s_piN_cex=s_piN_cex, s_conv=s_conv, s_NN_elastic=s_NN_elastic,
                                      s_NN_inelastic=s_NN_inelastic, f_NN_cex=f_NN_cex)
    wp = fsi_pion_reweight((record["bc"], record["sa"], record["ss_el"], record["ss"], record["si"],
                           record["pi_hh"], record["pi_a"], record["sa_c"], record["ss_el_c"],
                           record["ss_c"], record["si_c"], record["nh"]), sabs, s_el, s_cex, s_conv)
    nse = jnp.broadcast_to(jnp.asarray(sscat), (3,)) if s_NN_elastic is None else jnp.asarray(s_NN_elastic)
    nsi = jnp.broadcast_to(jnp.asarray(sscat), (3,)) if s_NN_inelastic is None else jnp.asarray(s_NN_inelastic)
    wn = fsi_nucleon_reweight((record["hh"], record["a"], record["iso"], record["finel"], record["inel"],
                              record["ns"]), nse, nsi)
    wc = fsi_nncex_reweight((record["hh"], record["iso"], record["inel"], record["swap"], record["ns"]),
                            f_NN_cex)                            # NN charge-exchange fraction (nominal 0.5)
    return wp * wn * wc


# ---- RAGGED (bank) record --------------------------------------------------------------------------- #
# The in-engine record MUST be dense (n, K): XLA needs static shapes inside the jitted walk.  But the tail
# slots are pure padding -- mean occupancy is 2.3/96 for pions and 10.6/64 for nucleons, i.e. the dense
# bank is ~97% zeros.  compact_fsi_record() drops the padding at WRITE time into flat (M,) slot arrays
# plus a per-slot event index (exactly how the bank already stores the ragged final state: fs_* + _eidx).
# 1.87M-event bank: ~8.4 GB dense -> ~2 GB ragged, and every Jacobian jvp touches 40x fewer slots.
_P_SLOT = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c")
_N_SLOT = ("hh", "a", "iso", "finel", "inel", "swap")


def compact_fsi_record(rec):
    """Dense (n, K) kind-1 record -> ragged: flat slot arrays + per-slot event index (p_eidx / n_eidx).
    Pure numpy (called once, at bank-write time).  Physics-preserving: it only drops the padding.
    FLAT record (from FLAT_FSI_REC streaming, carries gc_p/gc_n): already ragged -- just TRIM to the
    cursor (the tail past gc has eidx=n_events, dropped by the reweight anyway)."""
    import numpy as _np
    if "gc_p" in rec:                                            # already-flat streaming record: trim + pass
        gp = int(_np.asarray(rec["gc_p"])); gn = int(_np.asarray(rec["gc_n"]))
        Tp = len(_np.asarray(rec["bc"])); Tn = len(_np.asarray(rec["hh"]))
        # gc counts ALL interactions (incl. dropped-past-budget) -> gc > buffer means silent truncation.
        # Fail LOUD with the total budget needed (the flat budget is ~ n*E[interactions], tail-insensitive).
        if gp > Tp or gn > Tn:
            raise ValueError(f"FLAT FSI record overflow: pion {gp}/{Tp}, nucleon {gn}/{Tn}. Raise the flat "
                             f"TOTAL budget (rec_caps) -- it must be >= the batch's total interaction count.")
        out = {f: _np.asarray(rec[f])[:gp] for f in _P_SLOT}
        out.update({f: _np.asarray(rec[f])[:gn] for f in _N_SLOT})
        out["p_eidx"] = _np.asarray(rec["p_eidx"])[:gp]; out["n_eidx"] = _np.asarray(rec["n_eidx"])[:gn]
        return out
    out = {}
    for names, cnt, tag in ((_P_SLOT, "nh", "p"), (_N_SLOT, "ns", "n")):
        c = _np.asarray(rec[cnt])
        K = _np.asarray(rec[names[0]]).shape[1]
        cmax = int(c.max()) if c.size else 0
        # The in-cascade log is a fixed-width (n, K) buffer (JAX static shapes); K = rec_caps.  If any
        # particle took MORE steps than K, that buffer truncated it while `c` still reports the true count
        # -> the compaction below would be inconsistent.  Fail LOUD and say exactly what K is needed (the
        # bigger nuclei, e.g. Ar, log more steps than the carbon-tuned caps).  Only the transient buffer
        # grows with K; the stored ragged output does not.
        if cmax > K:
            raise ValueError(
                f"FSI '{tag}' record overflow: max per-event steps = {cmax} exceeds the buffer cap K = {K} "
                f"(rec_caps). Set rec_caps / ADONIS_REC_CAPS for this species above {cmax} (with margin).")
        keep = _np.arange(K)[None, :] < c[:, None]                # the used slots, per event
        out[f"{tag}_eidx"] = _np.repeat(_np.arange(len(c), dtype=_np.int32), c)
        for f in names:
            out[f] = _np.asarray(rec[f])[keep]
        assert len(out[f"{tag}_eidx"]) == int(keep.sum())
    return out


def pool_fsi_reweight_flat(record, sabs, sscat, *, s_piN_elastic=None, s_piN_cex=None, s_conv=1.0,
                           s_NN_elastic=None, s_NN_inelastic=None, f_NN_cex=0.5):
    """pool_fsi_reweight on a RAGGED record (compact_fsi_record output + n_events).  Same per-slot physics
    (cascade_discrete.*_slot_factor), ragged reduction.  record["n_events"] gives the event count."""
    R = record
    # OVERFLOW GUARD on the DIRECT (uncompacted) flat path: a raw flat record carries gc_p/gc_n = the TRUE
    # interaction count (incl. dropped-past-buffer).  If gc > buffer, the buffer is TRUNCATED and the reweight
    # would be silently wrong -- fail loud here too (not only in compact_fsi_record).  A compacted record has
    # no gc_* (already trimmed & guarded), so this is a no-op there.
    if "gc_p" in R:
        gp = int(jnp.asarray(R["gc_p"])); gn = int(jnp.asarray(R["gc_n"]))
        Tp = len(jnp.asarray(R["bc"])); Tn = len(jnp.asarray(R["hh"]))
        if gp > Tp or gn > Tn:
            raise ValueError(f"FLAT FSI record overflow on reweight: pion {gp}/{Tp}, nucleon {gn}/{Tn}. "
                             f"Raise the flat TOTAL budget (rec_caps) above the batch total interaction count.")
    n = int(R["n_events"])
    s_el = sscat if s_piN_elastic is None else s_piN_elastic
    s_cex = sscat if s_piN_cex is None else s_piN_cex
    wp = fsi_pion_reweight_flat((R["bc"], R["sa"], R["ss_el"], R["ss"], R["si"], R["pi_hh"], R["pi_a"],
                                 R["sa_c"], R["ss_el_c"], R["ss_c"], R["si_c"]),
                                R["p_eidx"], n, sabs, s_el, s_cex, s_conv)
    nse = jnp.broadcast_to(jnp.asarray(sscat), (3,)) if s_NN_elastic is None else jnp.asarray(s_NN_elastic)
    nsi = jnp.broadcast_to(jnp.asarray(sscat), (3,)) if s_NN_inelastic is None else jnp.asarray(s_NN_inelastic)
    wn = fsi_nucleon_reweight_flat((R["hh"], R["a"], R["iso"], R["finel"], R["inel"]),
                                   R["n_eidx"], n, nse, nsi)
    wc = fsi_nncex_reweight_flat((R["hh"], R["iso"], R["inel"], R["swap"]), R["n_eidx"], n, f_NN_cex)
    return wp * wn * wc


# SEGMENT-LOG fields written by the in-engine logger (see run_cascade_pool log_cap path).  One row per
# cascade SEGMENT (a particle's free-flight episode ending in an interaction or a terminal escape):
#   chan: 0 transmit/escape | pion {1 elastic,2 charge-ex,3 abs,4 conv} | nucleon {1 elastic,2 inelastic}
#   inc_pid/incp: the particle's pid + segment-start |p|;  track_id/parent_id/gen: provenance (G4-like);
#   cont_*: the continuing particle post-interaction;  n1/n2/pio: the ejected daughters (pid,|p|,alive).
_LOG_I = ("chan", "inc_pid", "parent_id", "gen", "track_id", "cont_pid", "n1_pid", "n1_al",
          "n2_pid", "n2_al", "pio_pid", "pio_al")
_LOG_F = ("incp", "cont_p", "n1_p", "n2_p", "pio_p")


def _take_rows(d, idx):
    """Gather rows (leading axis) of a pytree-of-arrays dict at integer index `idx`."""
    return {k: v[idx] for k, v in d.items()}


def _round_betamax(stk, wait, round_gt, betamax):
    """ACHILLES AdaptiveStep, P-invariantly: per-event beta_max = max beta over ALL the event's alive
    particles (stack UNION wait -- a P-INDEPENDENT set), SNAPSHOTTED once per gtime-round.  A new round is
    detected when the event's min(gtime) over alive particles advances past `round_gt`; at that instant the
    cascade's lock-step has brought every alive particle to that min gtime, so the max is the true round
    beta_max at ANY P (P only changes how the set is split between stack and wait).  Within a round the
    stored beta_max is reused (so an already-processed fast particle still counts).  Returns (round_gt',
    beta_max').  dt_evt = step/beta_max -> a particle ALONE has beta_max=its own beta -> moves a full step
    (no freeze, no arbitrary cap -- the same slow-particle protection ACHILLES gets for free)."""
    def _bg(b):
        be = jnp.linalg.norm(b["p4"][..., 1:], axis=-1) / jnp.clip(b["p4"][..., 0], 1e-9, None)
        return be, b["alive"], b["gtime"]
    be_s, al_s, gt_s = _bg(stk)
    be_w, al_w, gt_w = _bg(wait)
    be = jnp.concatenate([be_s, be_w], axis=1)
    al = jnp.concatenate([al_s, al_w], axis=1)
    gt = jnp.concatenate([gt_s, gt_w], axis=1)
    g_min = jnp.min(jnp.where(al, gt, jnp.int32(1 << 30)), axis=1)
    bmax_new = jnp.max(jnp.where(al, be, 0.0), axis=1)
    new_round = g_min > round_gt
    betamax = jnp.where(new_round, jnp.clip(bmax_new, 1e-6, None), betamax)
    round_gt = jnp.where(new_round, g_min, round_gt)
    return round_gt, betamax


def run_cascade_pool(init, stepper, key, state0, M, max_steps, M_out=24, prim_origin=-999,
                     rec_caps=None, log_cap=None, bg=None, q_cap=0,
                     pending=None, n_w=None, per_event_cap=None, time_sync=False, step=0.04):
    """POOLED engine loop (docs/logbook/cascade_pool_engine.md): ONE fixed-size (W, M) particle stack
    stepped once per step; the in/out reconcile (pool_reconcile) runs INSIDE the step.
      stepper(stack, key, state) -> (stack2 (W,M), terminal (W,M) bool, spawn (W,K), state2[, rec|seg])
    Escaped terminals (the cascade FINAL STATE) accumulate into a fixed (W, M_out) output batch.  RNG is
    PER EVENT (key folded by evt_id+nstep) so an event is reproducible regardless of slot/step.

    TWO modes share the per-step body `_apply_step` (single source of truth):
      * NO-REFILL (pending=None, default): the working set IS the n events, run lock-step until all done or
        `max_steps`.  Bit-identical to the pre-refill engine -- the path every existing caller uses.
      * REFILL (pending given): the working set holds `n_w` event-slots fed from a PENDING POOL of all
        N_total events; when a slot's event finishes (no live particle) OR hits its per-event step cap, its
        accumulators are FLUSHED into global (N_total,...) buffers at its evt_id and the slot is REFILLED
        from a cursor.  `per_event_cap` replaces the global max_steps.  Removes the lock-step waste (a
        single long event no longer makes all events step to max_steps) and keeps the (W,M) tensor small.
        The ONLY behavioural change vs no-refill is none: at n_w=N_total it is bit-exact (gate); at
        n_w<N_total the per-event outputs are identical (compare by evt_id), only occupancy/wall changes.
    `prim_origin` (RES): origin tag of the PRIMARY pion -> its terminal fate is latched per event.
    rec_caps=(Kp,Kn): accumulate the per-event kind-1 FSI reweight record.  log_cap=L: in-engine segment
    logger.  rec_caps and log_cap are mutually exclusive.
    Returns (out_batch, stack_overflow, out_overflow, prim_fate[, fsi_record | (log, counts, log_overflow)])."""
    with_rec = rec_caps is not None
    do_log = log_cap is not None
    assert not (with_rec and do_log), "rec_caps and log_cap are mutually exclusive"
    Kp, Kn = rec_caps if with_rec else (1, 1)
    L = int(log_cap) if do_log else 1                                # dummy (.,1) buffers when not logging
    Q = int(q_cap) if q_cap and q_cap > 0 else 0                     # particle waiting-buffer width (0 = drop)
    cap = int(per_event_cap) if per_event_cap else int(max_steps)    # per-event step cap (refill mode)

    def _logbuf(W):
        return {**{k: jnp.zeros((W, L), jnp.int32) for k in _LOG_I},
                **{k: jnp.zeros((W, L), jnp.float32) for k in _LOG_F}}

    def _apply_step(stk, state, kk, step_i, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bg_w, ar,
                    round_gt, betamax, evt_id=None):
        """ONE pooled step on the working set + accumulation (out, prim latch, rec, seg log) + reconcile.
        Scalars sofl/oofl/rofl/logofl accumulate globally; out/prim/rb/log/wptr are per-slot.  Returns the
        updated working-set state; identical maths regardless of refill mode.  time_sync: ACHILLES
        AdaptiveStep per-event dt = step/beta_max (snapshotted per gtime-round over stack UNION wait)."""
        if time_sync:
            round_gt, betamax = _round_betamax(stk, wait, round_gt, betamax)
            # DECOUPLED diagnostic (ADONIS_DECOUPLED=1): fixed Dt=step (each particle moves beta*step from
            # its OWN beta, no beta_max coupling) instead of ACHILLES adaptive Dt=step/beta_max.  Same
            # beta_i:beta_j ratios, different absolute step -> isolates "coupled vs decoupled time-sync".
            dt_evt = jnp.full_like(betamax, step) if os.environ.get("ADONIS_DECOUPLED") == "1" else step / betamax
        else:
            dt_evt = None
        # step_i is passed in EVERY path now (not just logging): track_id (creation index, used by the
        # within-cohort processing order) needs the real step counter -- a stale step=0 would collapse all
        # daughters to the same id and break the ACHILLES-order tiebreak.
        if bg_w is not None:
            _step = stepper(stk, kk, state, step_i, bg=bg_w, dt_evt=dt_evt)
        else:
            _step = stepper(stk, kk, state, step_i, dt_evt=dt_evt)
        stk2, terminal, spawn, state2 = _step[:4]
        extra = _step[4] if len(_step) >= 5 else None
        term_batch = {**stk2, "alive": terminal}
        out2, oo = compact({k: jnp.concatenate([out[k], term_batch[k]], axis=1) for k in out}, M_out)
        # Latch the PRIMARY's terminal fate.  Was PION-only (RES primary pion); now species-agnostic so a
        # nucleon-beam primary's fate (ESCAPE or CAPTURE) is also latched -- only the continuing primary
        # carries origin==prim_origin, so this catches exactly it (pion for RES, nucleon for a beam; T2K
        # QE has prim_origin=-999 -> no match, unchanged).
        isprim = (stk2["origin"] == prim_origin) & (stk2["fate"] != FATE_NONE)
        anyp = jnp.any(isprim, axis=1); j = jnp.argmax(isprim, axis=1)
        prim = jnp.where(anyp & (prim == FATE_NONE), stk2["fate"][ar, j], prim)
        if with_rec:
            rec = extra
            pion_vals = [rec["pi_bc"], rec["pi_sa"], rec["pi_ss_el"], rec["pi_ss"], rec["pi_si"],
                         rec["pi_hh"], rec["pi_a"], rec["pi_sa_c"], rec["pi_ss_el_c"], rec["pi_ss_c"],
                         rec["pi_si_c"]]                                   # in _P_SLOT order
            nuc_vals = [rec["nu_hh"], rec["nu_a"], rec["nu_iso"], rec["nu_finel"], rec["nu_inel"],
                        rec["nu_swap"]]                                    # in _N_SLOT order
            if "gc_p" in rb:                                              # FLAT/STREAMING record (tail-free)
                new_p, gc_p, p_eidx, op = _rec_scatter_flat(
                    [rb[f] for f in _P_SLOT], rb["gc_p"], rb["p_eidx"], rec["pi_m"], pion_vals, evt_id, Kp, _NEVT)
                new_n, gc_n, n_eidx, on = _rec_scatter_flat(
                    [rb[f] for f in _N_SLOT], rb["gc_n"], rb["n_eidx"], rec["nu_m"], nuc_vals, evt_id, Kn, _NEVT)
                rb = {**dict(zip(_P_SLOT, new_p)), **dict(zip(_N_SLOT, new_n)),
                      "gc_p": gc_p, "gc_n": gc_n, "p_eidx": p_eidx, "n_eidx": n_eidx}
            else:                                                        # DENSE per-event (n, K) record
                new_p, nh, op = _rec_scatter([rb[f] for f in _P_SLOT], rb["nh"], rec["pi_m"], pion_vals, Kp)
                new_n, ns, on = _rec_scatter([rb[f] for f in _N_SLOT], rb["ns"], rec["nu_m"], nuc_vals, Kn)
                rb = {**dict(zip(_P_SLOT, new_p)), **dict(zip(_N_SLOT, new_n)), "nh": nh, "ns": ns}
            rofl = (rofl + op + on).astype(rofl.dtype)
        if do_log:
            seg = extra
            ev = (seg["chan"] > 0) | terminal                        # interactions (1-4) + escapes (0)
            evi = ev.astype(jnp.int32)
            off = jnp.cumsum(evi, axis=1) - evi                      # exclusive prefix -> per-event slot order
            tgt0 = wptr[:, None] + off
            logofl = logofl + (ev & (tgt0 >= L)).sum().astype(logofl.dtype)
            tgt = jnp.where(ev & (tgt0 < L), tgt0, L)                # OOB / non-events dropped by mode='drop'
            log = dict(log)
            for k in _LOG_I + _LOG_F:
                log[k] = log[k].at[ar[:, None], tgt].set(seg[k].astype(log[k].dtype), mode="drop")
            wptr = wptr + evi.sum(axis=1).astype(wptr.dtype)
        if Q > 0:                                                    # particle waiting-queue: keep overflow
            newstk, newwait, so = pool_reconcile(stk2, terminal, spawn, M, wait, Q)
        else:                                                        # drop overflow (wait stays dummy)
            newstk, _nw, so = pool_reconcile(stk2, terminal, spawn, M)
            newwait = wait
        sofl = sofl + so.astype(sofl.dtype); oofl = oofl + oo.astype(oofl.dtype)
        return newstk, state2, newwait, out2, prim, rb, log, wptr, sofl, oofl, rofl, logofl, round_gt, betamax

    # ---------------- NO-REFILL: the working set IS the n events (bit-exact to the pre-refill engine) ----
    if pending is None:
        n = init["alive"].shape[0]; ar = jnp.arange(n)
        _NEVT = n                                        # event count for the flat record's eidx range
        # INITIAL overflow (more primaries than M, e.g. RES pion+recoil at M=1) must go to the wait QUEUE,
        # not be dropped -- else the engine is not P-invariant (the initial compact silently discarded the
        # 2nd primary at small M).  Mirror pool_reconcile: pack init into M active + Q waiting.
        if Q > 0:
            _full, _ = compact(init, M + Q, sort_priority=True)
            stack = {k: v[:, :M] for k, v in _full.items()}
            wait0 = {k: v[:, M:M + Q] for k, v in _full.items()}
        else:
            stack, _ = compact(init, M, sort_priority=True)
            wait0 = empty_batch(n, max(Q, 1))
        out0 = empty_batch(n, M_out)
        rec0 = _empty_flat_fsi_record(Kp, Kn, n) if (with_rec and FLAT_FSI_REC) else _empty_fsi_record(n, Kp, Kn)
        log0 = _logbuf(n); wptr0 = jnp.zeros(n, jnp.int32); logofl0 = jnp.int32(0)
        evt_id0 = jnp.arange(n, dtype=jnp.int32); nstep0 = jnp.zeros(n, jnp.int32)

        # The per-particle caps (stepper: physical lpath budget + lstep safety backstop) guarantee every
        # particle terminates, so the loop normally ends via 'no particle alive' (early-exit) at ANY P.  The
        # _HARD_STEPS ceiling is a pure runaway tripwire: reaching it means a particle never terminated, and
        # run_cascade_pool RAISES below rather than silently truncating (-> visible anomaly, per design).
        def cond(st):
            return (st[0] < _HARD_STEPS) & (jnp.any(st[1]["alive"]) | jnp.any(st[12]["alive"]))

        def body(st):
            (i, stk, state, out, sofl, oofl, prim, rb, rofl, log, wptr, logofl, wait, evt_id, nstep,
             round_gt, betamax) = st
            kk = jax.vmap(lambda e, s: jax.random.fold_in(jax.random.fold_in(key, e), s))(evt_id, nstep)
            (newstk, state2, newwait, out2, prim2, rb2, log2, wptr2, sofl2, oofl2, rofl2, logofl2,
             round_gt2, betamax2) = _apply_step(
                stk, state, kk, i, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bg, ar,
                round_gt, betamax, evt_id=evt_id)
            return (i + jnp.int32(1), newstk, state2, out2, sofl2, oofl2, prim2, rb2, rofl2,
                    log2, wptr2, logofl2, newwait, evt_id, nstep + jnp.int32(1), round_gt2, betamax2)

        init_st = (jnp.int32(0), stack, state0, out0, jnp.int32(0), jnp.int32(0),
                   jnp.full(n, FATE_NONE, jnp.int32), rec0, jnp.int32(0), log0, wptr0, logofl0, wait0,
                   evt_id0, nstep0, jnp.full(n, -1, jnp.int32), jnp.ones(n))
        (_i_fin, stack, _, out, sofl, oofl, prim, rb, rofl, log, wptr, logofl, _wait,
         _evt, _ns, _rg, _bm) = jax.lax.while_loop(cond, body, init_st)
        _runaway = jnp.where(_i_fin >= _HARD_STEPS,
                             jnp.sum((jnp.any(stack["alive"], axis=1) | jnp.any(_wait["alive"], axis=1))
                                     .astype(jnp.int32)), jnp.int32(0))
        _raise_if_runaway(_runaway, "lock-step")
        if do_log:
            return out, sofl, oofl, prim, (log, wptr, logofl)
        if with_rec:
            return out, sofl, oofl, prim, (rb, rofl)
        return out, sofl, oofl, prim

    # ---------------- REFILL: working set of n_w slots fed from the pending pool of N_total events -------
    Ntot = pending["stack"]["alive"].shape[0]
    W = min(int(n_w) if n_w else Ntot, Ntot); ar = jnp.arange(W)
    _NEVT = Ntot                                     # event count for the flat record's eidx range
    _flat = with_rec and FLAT_FSI_REC                # streaming: rb IS the ONE global flat buffer (no grb flush)
    # INITIAL overflow (events with >M primaries, e.g. RES pion+recoil at M=1) must ride in the WAIT queue,
    # not be dropped -- mirror the no-refill init.  pstack = M active per event, pwait = Q waiting per event.
    # (Without this the RES recoil nucleon is silently dropped at refill setup -> N(p) halved.)
    if Q > 0:
        _pfull, _ = compact(pending["stack"], M + Q, sort_priority=True)
        pstack = {k: v[:, :M] for k, v in _pfull.items()}
        pwait = {k: v[:, M:M + Q] for k, v in _pfull.items()}
    else:
        pstack, _ = compact(pending["stack"], M, sort_priority=True)
        pwait = empty_batch(Ntot, 1)
    pbg = (pending["npos"], pending["nmom"], pending["nisp"])
    pcons = pending["consumed0"]
    # global per-event buffers (filled by flush-on-finish, indexed by evt_id)
    g_out = empty_batch(Ntot, M_out); g_prim = jnp.full(Ntot, FATE_NONE, jnp.int32)
    g_rb = {} if _flat else _empty_fsi_record(Ntot, Kp, Kn)   # flat: unused (rb is the global buffer)
    g_log = _logbuf(Ntot); g_wptr = jnp.zeros(Ntot, jnp.int32)
    # initial working set = first W events
    idx0 = jnp.arange(W, dtype=jnp.int32)
    stk0 = _take_rows(pstack, idx0); cons0 = pcons[idx0]
    bg0 = (pbg[0][idx0], pbg[1][idx0], pbg[2][idx0])
    out0 = empty_batch(W, M_out)
    rb0 = _empty_flat_fsi_record(Kp, Kn, Ntot) if _flat else _empty_fsi_record(W, Kp, Kn)
    log0 = _logbuf(W); wptr0 = jnp.zeros(W, jnp.int32)
    prim0 = jnp.full(W, FATE_NONE, jnp.int32)
    wait0 = _take_rows(pwait, idx0) if Q > 0 else empty_batch(W, max(Q, 1))   # recoil rides in wait from t=0
    cursor0 = jnp.int32(W); evt0 = idx0; nstep0 = jnp.zeros(W, jnp.int32)

    def rcond(st):
        (cursor, stk, cons, bgw, evt, nstep, wait, out, prim, rb, log, wptr,
         sofl, oofl, rofl, logofl, go, gp, grb, glog, gwp, round_gt, betamax, forced) = st
        return (cursor < Ntot) | jnp.any(stk["alive"]) | jnp.any(wait["alive"])

    def rbody(st):
        (cursor, stk, cons, bgw, evt, nstep, wait, out, prim, rb, log, wptr,
         sofl, oofl, rofl, logofl, go, gp, grb, glog, gwp, round_gt, betamax, forced) = st
        kk = jax.vmap(lambda e, s: jax.random.fold_in(jax.random.fold_in(key, e), s))(jnp.maximum(evt, 0), nstep)
        (newstk, cons2, newwait, out2, prim2, rb2, log2, wptr2, sofl2, oofl2, rofl2, logofl2,
         round_gt2, betamax2) = _apply_step(
            stk, cons, kk, nstep, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bgw, ar,
            round_gt, betamax, evt_id=evt)
        nstep2 = nstep + jnp.int32(1)
        still_alive = jnp.any(newstk["alive"], axis=1) | jnp.any(newwait["alive"], axis=1)
        hit_cap = (nstep2 >= cap)                                     # _HARD_STEPS per-event ceiling
        forced = (forced + jnp.sum((hit_cap & still_alive & (evt >= 0)).astype(jnp.int32))).astype(jnp.int32)
        finished = (~still_alive) | hit_cap                          # runaway tripwire -> _raise_if_runaway
        flush = finished & (evt >= 0)                                 # only real (non-idle) slots flush
        # FLUSH: scatter per-slot accumulators to the global buffers at evt_id (idle/non-finished -> Ntot, dropped)
        gi = jnp.where(flush, evt, Ntot)
        go = {k: go[k].at[gi].set(out2[k], mode="drop") for k in go}
        gp = gp.at[gi].set(prim2, mode="drop")
        if not _flat:                                                # flat: rb2 already streamed globally; no flush
            grb = {k: grb[k].at[gi].set(rb2[k], mode="drop") for k in grb}
        glog = {k: glog[k].at[gi].set(log2[k], mode="drop") for k in glog}
        gwp = gwp.at[gi].set(wptr2, mode="drop")
        # REFILL flushed slots from the cursor (creation order); idle slots (evt<0) are skipped
        rank = jnp.cumsum(flush.astype(jnp.int32)) - flush.astype(jnp.int32)
        new_id = (cursor + rank).astype(jnp.int32)
        take = flush & (new_id < Ntot)                                # this slot gets a fresh event
        cursor2 = (cursor + jnp.sum(take.astype(jnp.int32))).astype(jnp.int32)
        gidx = jnp.where(take, new_id, 0)                             # safe gather index
        rstk = _take_rows(pstack, gidx); rbg = (pbg[0][gidx], pbg[1][gidx], pbg[2][gidx]); rcons = pcons[gidx]
        rwait = _take_rows(pwait, gidx) if Q > 0 else empty_batch(W, max(Q, 1))   # fresh event's WAIT (its recoil)
        def _sel(new, old, mask, md=None):                            # per-slot pick (mask along axis 0)
            m = mask.reshape((-1,) + (1,) * (old.ndim - 1))
            return jnp.where(m, new, old)
        idle = finished & (~take)                                     # finished but no pending left -> go dead
        stk3 = {k: _sel(rstk[k], newstk[k], take) for k in newstk}
        stk3["alive"] = jnp.where(idle[:, None], False, stk3["alive"])  # drop any cap-cutoff survivors
        bg3 = tuple(_sel(rbg[t], bgw[t], take) for t in range(3))
        cons3 = _sel(rcons, cons2, take)
        # wait: refilled slot -> fresh event's wait (carries its recoil); finished+idle -> empty; else keep.
        _ewait = empty_batch(W, max(Q, 1))
        wait3 = {k: _sel(rwait[k], _sel(_ewait[k], newwait[k], finished), take) for k in newwait}
        # reset per-slot accumulators on EVERY finished slot (taken or now-idle); evt/nstep updated
        e_out = empty_batch(W, M_out); e_log = _logbuf(W)
        out3 = {k: _sel(e_out[k], out2[k], finished) for k in out2}
        if _flat:
            rb3 = rb2                                    # ONE global flat buffer streamed in-place: no per-slot reset
        else:
            e_rb = _empty_fsi_record(W, Kp, Kn)
            rb3 = {k: _sel(e_rb[k], rb2[k], finished) for k in rb2}
        log3 = {k: _sel(e_log[k], log2[k], finished) for k in log2}
        prim3 = jnp.where(finished, FATE_NONE, prim2)
        wptr3 = jnp.where(finished, 0, wptr2)
        evt3 = jnp.where(finished, jnp.where(take, new_id, jnp.int32(-1)), evt)
        nstep3 = jnp.where(finished, jnp.int32(0), nstep2)
        # reset the per-event time-sync snapshot on EVERY finished slot (a refilled slot is a NEW event,
        # so its first step must recompute beta_max): round_gt -> -1 triggers new_round next step.
        round_gt3 = jnp.where(finished, jnp.int32(-1), round_gt2)
        betamax3 = jnp.where(finished, 1.0, betamax2)
        return (cursor2, stk3, cons3, bg3, evt3, nstep3, wait3, out3, prim3, rb3, log3, wptr3,
                sofl2, oofl2, rofl2, logofl2, go, gp, grb, glog, gwp, round_gt3, betamax3, forced)

    init_st = (cursor0, stk0, cons0, bg0, evt0, nstep0, wait0, out0, prim0, rb0, log0, wptr0,
               jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0), g_out, g_prim, g_rb, g_log, g_wptr,
               jnp.full(W, -1, jnp.int32), jnp.ones(W), jnp.int32(0))
    out_st = jax.lax.while_loop(rcond, rbody, init_st)
    (_, _, _, _, _, _, _, _, _, _rb_fin, _, _, sofl, oofl, rofl, logofl, go, gp, grb, glog, gwp,
     _rg, _bm, _forced) = out_st
    _raise_if_runaway(_forced, "refill")
    if do_log:
        return go, sofl, oofl, gp, (glog, gwp, logofl)
    if with_rec:                                         # flat: rb IS the global ragged record; dense: grb
        return go, sofl, oofl, gp, (_rb_fin if _flat else grb, rofl)
    return go, sofl, oofl, gp


def _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, rec_caps=None, log_cap=None,
                  n_w=0, q_cap=0, per_event_cap=None):
    """POOLED-engine realization of cascade_nucleus (QE + RES), mapping the flat (n,M_out) terminal
    buffer back to the rich (pterm, nterms, overflow, created) schema.
    log_cap=L: run the SAME cascade with the in-engine SEGMENT logger on (with_seg stepper +
    run_cascade_pool log_cap) and return (log, counts, log_overflow) directly -- the single entry point
    for the cascade-vertex/segment matrix (gen_cascade_segments), so there is NO duplicated cascade.
      QE : gen-0 stack = the struck->proton (1 NUCLEON slot); pterm = QE "none"; created = leading
           surviving pion.
      RES: gen-0 stack = the PRIMARY pion (PION slot, origin-tagged) + the RES recoil nucleon; the pion's
           own scatter-recoils / absorption protons / created pions spawn natively during the walk.
           pterm = the primary pion's outcome (escape -> its pid/p4; absorbed -> pid 0; converted -> -1,
           via the latched fate); created = leading surviving NON-primary pion."""
    ar = jnp.arange(n)
    if channel == "qe":
        g0 = empty_batch(n, 1)
        g0["alive"] = jnp.ones((n, 1), bool)
        g0["species"] = jnp.full((n, 1), NUCLEON, jnp.int32)
        g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]
        g0["p4"] = p_N[:, None, :]; g0["pos"] = su["pos0"][:, None, :]
        g0["track_id"] = jnp.zeros((n, 1), jnp.int32)                  # primary id (inert unless logging)
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))   # (n,2) per-event base key
        g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]  # primary idx 0 (n,1,2)
        prim_origin = -999
    else:                                                              # RES: primary pion + recoil nucleon
        g0 = empty_batch(n, 2)
        g0["alive"] = jnp.ones((n, 2), bool)
        g0["species"] = jnp.array([PION, NUCLEON], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
        g0["p4"] = jnp.stack([p_pi, p_N], axis=1)
        g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (n, 2, 3))
        g0["origin"] = jnp.array([_ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["track_id"] = jnp.array([0, 1], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)  # pion=0, recoil=1
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))   # (n,2) per-event base key
        g0["pkey"] = jnp.stack([jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base),   # pion primary idx 0
                                jax.vmap(lambda b: jax.random.fold_in(b, 1))(_base)], axis=1)  # recoil idx 1 (n,2,2)
        prim_origin = _ORIG_PRIM_PI
    # n_w>0 (refill): run the persistent-refill engine -- a working set of n_w event-slots fed from the
    # pending pool of all n events (g0 + per-event nucleus background).  n_w=0 -> lock-step.  q_cap>0
    # enables the particle waiting-queue in EITHER path (keeps overflow particles).
    pend = dict(stack=g0, consumed0=su["consumed0"], npos=su["npos"], nmom=su["nmom"],
                nisp=su["nisp"]) if n_w and n_w > 0 else None
    nw = n_w if pend is not None else None
    if log_cap is not None:                                            # SEGMENT-LOGGER path: same cascade, logger on
        stepper = make_pool_stepper(su, cfg, with_seg=True)
        _o, _so, _oo, _pf, logtuple = run_cascade_pool(
            g0, stepper, knuc, su["consumed0"], M=1, max_steps=cfg.max_steps, M_out=24,
            prim_origin=prim_origin, log_cap=log_cap, pending=pend, n_w=nw, q_cap=q_cap,
            per_event_cap=per_event_cap, time_sync=cfg.time_step, step=cfg.step)
        return logtuple                                               # (log dict, counts (n,), log_overflow)
    stepper = make_pool_stepper(su, cfg, with_rec=rec_caps is not None)
    _rc = run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=1, max_steps=cfg.max_steps,
                           M_out=24, prim_origin=prim_origin, rec_caps=rec_caps, pending=pend, n_w=nw,
                           q_cap=q_cap, per_event_cap=per_event_cap, time_sync=cfg.time_step, step=cfg.step)
    if rec_caps is not None:
        out, sofl, oofl, prim_fate, (fsi_rec, _rofl) = _rc      # joint per-event kind-1 FSI record
    else:
        out, sofl, oofl, prim_fate = _rc; fsi_rec = None
    sp = out["species"]; chg = out["charge"]; al = out["alive"]; p4o = out["p4"]
    nterms = [dict(species=sp, pid=jnp.where((sp == NUCLEON) & (chg == 1), 2212, 2112),
                   charge=chg,                                       # raw charge idx: nucleon isospin
                                                                    # (1=p) | pion charge (0:+,1:0,2:-).
                                                                    # ADDITIVE -- pid unchanged (golden-safe);
                                                                    # lets consumers count pions by charge.
                   p4=p4o, alive=al, origin=out["origin"], gen=out["gen"], nsc=out["nsc"])]  # nsc: uniform
    #   cascade-outcome record needs the per-particle scatter count (additive; existing consumers key-access)
    is_surv_pi = (sp == PION) & al                                     # escaped pions (output is escape-only)
    if channel == "qe":
        pim = jnp.linalg.norm(p4o[:, :, 1:], axis=2) * is_surv_pi
        jpi = jnp.argmax(pim, axis=1); has_pi = pim[ar, jpi] > 0.0
        created = dict(pid=jnp.where(has_pi, _CH_PID[chg[ar, jpi]], 0), p4=p4o[ar, jpi],
                       w=jnp.ones((n,)), alive=has_pi)
        pterm = dict(species=jnp.zeros((n,), jnp.int32), pid=jnp.zeros((n,), jnp.int32),
                     p4=jnp.zeros((n, 4)), charge=jnp.zeros((n,), jnp.int32), w=jnp.ones((n,)),
                     alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
        return pterm, nterms, sofl + oofl, created, fsi_rec, prim_fate
    # RES: split surviving pions into primary (origin tag) vs created
    is_prim = is_surv_pi & (out["origin"] == _ORIG_PRIM_PI)            # escaped primary pion (>=0 per event)
    jp = jnp.argmax(is_prim, axis=1); esc_prim = jnp.any(is_prim, axis=1)
    prim_ch = chg[ar, jp]
    # pterm pid: escaped -> charge->pid; converted -> -1 (vetoes); absorbed/none -> 0.
    pterm_pid = jnp.where(prim_fate == FATE_ESCAPE, _CH_PID[prim_ch],
                jnp.where(prim_fate == FATE_CONVERT, -1, 0)).astype(jnp.int32)
    pterm = dict(species=jnp.full((n,), PION, jnp.int32), pid=pterm_pid,
                 p4=jnp.where(esc_prim[:, None], p4o[ar, jp], 0.0), charge=prim_ch,
                 w=jnp.ones((n,)), alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
    is_cr = is_surv_pi & (out["origin"] != _ORIG_PRIM_PI)             # cascade-created surviving pion
    cim = jnp.linalg.norm(p4o[:, :, 1:], axis=2) * is_cr
    jc = jnp.argmax(cim, axis=1); has_cr = cim[ar, jc] > 0.0
    created = dict(pid=jnp.where(has_cr, _CH_PID[chg[ar, jc]], 0), p4=p4o[ar, jc],
                   w=jnp.ones((n,)), alive=has_cr)
    return pterm, nterms, sofl + oofl, created, fsi_rec, prim_fate


def cascade_nucleus(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, sabs=1.0, sscat=1.0,
                      channel="res", rec_caps=None, log_cap=None, n_w=None, q_cap=None, per_event_cap=None,
                      su_external=None, return_fate=False):
    """Faithful engine, SHARED by RES (CC1pi) and QE (CC0pi).
    channel="res": a primary pion segment (+ its top-K knockouts) then a NUCLEON BFS over {RES recoil,
                   pion knockouts}; pterm = the surviving pion.
    channel="qe":  NO primary pion -- gen-0 nucleon = the QE proton (p_N); pterm = "no pion" (pid 0).
    Both share the nucleon BFS (top-K knockouts) + the created-pion (NN->NDelta->Npi) re-entry, so the
    meson veto (no surviving pion for CC0pi / exactly one pi+ for CC1pi) is handled uniformly.
    rec_caps=(Kp,Kn) (pool only): also return the joint per-event kind-1 FSI reweight record as a 5th
    element (for pool_fsi_reweight / the differentiable blueprint).  Default None -> 4-tuple as before.
    log_cap=L: run the SAME cascade with the in-engine SEGMENT logger and return (log, counts, overflow)
    -- the single entry point for the cascade-vertex/segment matrix (no duplicated cascade).
    ENGINE DEFAULTS (refill + waiting-queue ON for every consumer):
      n_w   : None -> refill with working set min(_DEFAULT_NW, n);  0 -> lock-step (bit-exact reference);
              int>0 -> refill with min(n_w, n).
      q_cap : None -> _DEFAULT_QCAP (keep overflow particles, sofl->0);  0 -> legacy drop-on-overflow.
      per_event_cap: None -> cfg.max_steps (per-slot step cap in the refill path).
    Returns (pterm, nucleon_terminals_per_gen, overflow, created[, fsi_record]) | (log, counts, overflow)."""
    # su_external (ablation harness): run on an EXTERNALLY supplied nucleus (e.g. ACHILLES's exact per-event
    # background + struck vertex) instead of sampling our own -- isolates transport from input generation.
    # Must provide npos,nmom,nisp,pos0,consumed0,ch0,kp (same keys setup_nucleus returns).
    su = setup_nucleus(p_pi, pid_pi, pid_Ni, cfg, key) if su_external is None else su_external
    n = p_pi.shape[0]
    _kpi, knuc, _kpi2 = jax.random.split(su["kp"], 3)     # knuc drives the pool (RNG stream preserved)
    # Resolve engine defaults: refill ON (n_w window) + waiting-queue ON (q_cap) for ALL consumers.
    nw_eff = 0 if n_w == 0 else (min(_DEFAULT_NW, n) if n_w is None else min(int(n_w), n))
    q_eff = _DEFAULT_QCAP if q_cap is None else int(q_cap)
    # POOLED engine (the single cascade core): ONE fixed-size stack stepped once/step, in/out reconcile
    # inside the step; persistent-refill working set + keep-overflow queue by default (n_w=0 -> lock-step).
    # Validated vs ACHILLES + bit-exact gates (docs/logbook/cascade_persistent_refill_plan.md).
    if log_cap is not None:
        return _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, log_cap=log_cap,
                             n_w=nw_eff, q_cap=q_eff, per_event_cap=per_event_cap)
    res6 = _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, rec_caps=rec_caps,
                         n_w=nw_eff, q_cap=q_eff, per_event_cap=per_event_cap)
    # res6 = (pterm, nterms, overflow, created, fsi_rec, prim_fate).  Default: the legacy 5-/4-tuple
    # (fsi_rec only when rec_caps).  return_fate=True appends prim_fate (the uniform cascade-outcome
    # driver uses it to derive reacted/absorbed for ANY probe, not just tagged beams).
    base = res6[:5] if rec_caps is not None else res6[:4]
    return (base + (res6[5],)) if return_fate else base


# ---------------------------------------------------------------------------- #
# Unified public API.
DiscreteCascadeConfig = CascadeConfig    # backward-compat alias (deprecated; use CascadeConfig)

# Jitted production entry: config/channel/shape args are static, arrays + key traced.
cascade_nucleus_jit = jax.jit(
    cascade_nucleus,
    static_argnames=("cfg", "channel", "rec_caps", "log_cap", "n_w", "q_cap", "per_event_cap"),
)
