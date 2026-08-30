"""Jitted, differentiable intranuclear cascade (ADoNIS FSI).

Pion + nucleon + secondary transport over a shared nucleus, pooled (M=1 active slot +
FIFO wait queue + persistent event refill), differentiable via kind-1 reweighting, and
jit-able end to end.

Cross-section source-of-truth stays in the imported libs (pion_nuclear_xsec,
interactions.meson_baryon_xsec, nn_inelastic, absorption_modes). Public entry:
cascade_nucleus (eager core) and cascade_nucleus_jit (jitted). Config: CascadeConfig
(DiscreteCascadeConfig = alias).
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

from adonis.fsi import pion_nuclear_xsec as ox
from adonis.fsi.interactions import meson_baryon_xsec
from adonis.fsi.absorption_modes import kernel_tables as _abs_kernel_tables
from adonis.fsi import nn_inelastic as nni
from adonis.constants import mp as _MP_PHYS, mn as _MN_PHYS

HBARC = ox.HBARC
M_N = ox.M_N
MB_TO_FM2 = 0.1
_PI_MASS = {211: ox.M_PIP, 111: ox.M_PI0, -211: ox.M_PIP}
_DENS = {}


def _read_density_file(name):
    """(r, rho) from an ACHILLES number-density table, resolved under the ACHILLES data directory."""
    from adonis.io import require
    d = np.loadtxt(require("densities", name), comments="#")
    return d[:, 0], d[:, 1]


def _load_density(name="c12.prova.txt", name_n=None):
    """Proton & neutron number densities.  ACHILLES reads SEPARATE p/n densities (Nucleus.cc:36-80);
    for N=Z nuclei (C) name_n defaults to name -> rho_n == rho_p.  rho_n is interpolated onto the
    proton radial grid so a single rgrid serves both species.  Radius = first r where rho_proton <
    1e-6 fm^-3 (ACHILLES minDensity, Nucleus.cc:49-51).  Returns (rgrid, rho_p, rho_n, radius)."""
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
    """Number density of one species at radius r [fm], interpolated; 0 beyond the grid.  Caller
    passes the proton OR neutron density array (rho_p / rho_n)."""
    return jnp.interp(r, rgrid, rho, left=rho[0], right=0.0)


def _kf_local(rho_species):
    """Local Fermi momentum [MeV]: k_F = cbrt(rho * 3 pi^2) * hbarc (ACHILLES Local FG)."""
    return jnp.cbrt(jnp.clip(rho_species, 0.0, None) * 3.0 * jnp.pi ** 2) * HBARC


def _two_body_cm_scatter(p_pi, p_N, m_out_pi, key, cos_cm=None, m_recoil=None):
    """Elastic/charge-exchange piN -> pi'N' (or NN -> N'N') two-body kinematics.  cos_cm is the CM
    scattering angle, measured from the incoming-particle CM direction (ACHILLES convention); sampled
    from the DCC angular CDF by the caller, or isotropic if None.  Returns the outgoing (m_out_pi)
    4-momentum in the lab.  m_recoil defaults to avg M_N; pass the physical per-species mass so the 2->2
    final-state masses match ACHILLES GenerateMomentum."""
    P = p_pi + p_N
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
    beta = P[1:] / P[0]
    zaxis = _boost(p_pi, -beta)[1:]
    zhat = zaxis / jnp.clip(jnp.linalg.norm(zaxis), 1e-9, None)
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


_M_ETA_PHYS = 547.862
_CH_PID = jnp.array([211, 111, -211, 221])
_CH_MASS = jnp.array([ox.M_PIP, ox.M_PI0, ox.M_PIP, _M_ETA_PHYS])
_QE_IN = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
_QE_OUT = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

def sample_vertex(key, n, nucleus="c12.prova.txt", density_n=None):
    """Sample n production vertices in the nucleus ~ rho(r) (radial pdf rho(r) r^2), as the
    pion's cascade starting point.  Uses the proton density grid (== total shape for N=Z)."""
    rgrid, rho, _rho_n, radius = _load_density(nucleus, density_n)
    rg = np.asarray(rgrid); rh = np.asarray(rho)
    rr = np.linspace(0.0, float(radius), 600)
    pdf = np.interp(rr, rg, rh, left=rh[0], right=0.0) * rr ** 2
    pdf = pdf / pdf.sum()
    kr, kd = jax.random.split(key)
    cdf = jnp.cumsum(jnp.asarray(pdf))
    u = jax.random.uniform(kr, (n,))
    idx = jnp.clip(jnp.searchsorted(cdf, u), 0, len(rr) - 1)
    rad = jnp.asarray(rr)[idx]
    dirs = jax.random.normal(kd, (n, 3)); dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs * rad[:, None]


M_N_GEV = M_N / 1000.0

def nn_elastic_sigma(sqrts_mev, same_iso, mn_gev=M_N_GEV):
    """NN elastic cross section [mb] (ACHILLES NNElastic, GiBUU).  sqrts in MeV; `same_iso` True for
    pp/nn, False for np.  Piecewise in p_lab [GeV].  `mn_gev` is the per-pair average mass (ACHILLES
    NNElastic.cc:184 uses the actual pair mass in threshold/plab; default = global avg)."""
    sqrts = sqrts_mev / 1000.0
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


_ABS_W_NP, _ABS_PART_NP = _abs_kernel_tables()

M_N = ox.M_N
from adonis.constants import mp as _MP_PHYS, mn as _MN_PHYS
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
    nmax=36000 = all configs in QMC_configs.out.gz (ACHILLES uses the full set)."""
    if name not in _CFG:
        from adonis.io import require
        path = require("data", "configurations", name, home=True)
        with gzip.open(path, "rt") as f:
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


_KSLAB = 3


def pion_branch_reweight(brec, sabs, sscat):
    """Kind-1 branching reweight from compressed walk records brec =
    (branch (n,K) int {0 scatter, 1 abs, 2 conversion}, sa, ss, si (n,K), n_hits (n,)).
    Per-hit likelihood ratio p_branch(theta)/p_branch(nominal) with
    p_abs = sabs*sa/D, p_scat = sscat*ss/D, p_conv = si/D, D = sabs*sa + sscat*ss + si
    (conversion sigma unscaled).  Pure in (sabs, sscat); == in-propagation w_fsi; reduces to the
    two-branch formula when si = 0."""
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
    """Promote the survival pair (a, g) to a common dtype before exponentiating.

    The bank stores `a` as float32 while `g` inherits float64 from the knob vector; exponentiating
    in mismatched dtypes breaks bit-exactness of the nominal (g == 1) reweight.  Same-dtype exp
    keeps nominal exact."""
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

    The log-sum accumulates in the default float dtype (float64 under jax_enable_x64) even though
    the bank stores the slot sigmas as float32, to avoid losing precision relative to the dense
    float32 product."""
    acc = jnp.result_type(float)
    lp = jnp.log(jnp.clip(per, 1e-300, None)).astype(acc)
    return jnp.exp(jnp.zeros(n_events, acc).at[eidx].add(lp))


def pion_slot_factor(code, sa, ss_el, ss, si, hh, a, sa_c, ss_el_c, ss_c, si_c,
                     s_abs, s_el, s_cex, s_conv):
    """Per-slot pion FSI likelihood ratio -- branch x survival.  One slot = one in-slab candidate step
    (hit or not), the closest in-slab nucleon at that step.  A sigma scale moves two things:
      (1) BRANCH, given an interaction:  s_realized * D0/D,   D0 = sa+ss+si,
          D = s_abs*sa + s_el*ss_el + s_cex*(ss-ss_el) + s_conv*si   [at the struck candidate]
      (2) SURVIVAL -- whether it interacts.  The walk draws the hit with prob = exp(-a) per candidate
          (_pion_step), and sigma_tot -> g*sigma_tot maps a -> a/g, so with g = D_c/D0_c at the closest
          candidate:   hit: pk/p0    no-hit: (1-pk)/(1-p0),    p0 = exp(-a), pk = exp(-a/g).
    Under a common rescale the branch factor is 1 but survival is not (mean free path changes).
    == 1 at all-nominal (g = 1 -> pk = p0)."""
    hitf = hh.astype(bool)
    sa = jnp.where(hitf, sa, 1.0); ss_el = jnp.where(hitf, ss_el, 1.0)
    ss = jnp.where(hitf, ss, 1.0); si = jnp.where(hitf, si, 0.0)

    def _sigma_ratio(sa_, ss_el_, ss_, si_):
        """D/D0 = sum_i s_i f_i, written as 1 + sum_i (s_i - 1) f_i: algebraically identical, but exactly
        1 at nominal in any precision (every (s_i - 1) is 0) -- keeps the reweight bit-exact at nominal."""
        D0 = jnp.clip(sa_ + ss_ + si_, 1e-12, None)
        fcex = jnp.clip(ss_ - ss_el_, 0.0, None) / D0
        return (1.0 + (s_abs - 1.0) * (sa_ / D0) + (s_el - 1.0) * (ss_el_ / D0)
                + (s_cex - 1.0) * fcex + (s_conv - 1.0) * (si_ / D0))

    s_real = jnp.where(code == 0, s_el, jnp.where(code == 1, s_cex, jnp.where(code == 2, s_abs, s_conv)))
    branch = s_real / jnp.clip(_sigma_ratio(sa, ss_el, ss, si), 1e-12, None)
    g = jnp.clip(_sigma_ratio(sa_c, ss_el_c, ss_c, si_c), 1e-6, None)
    a, g = _match_dtype(a, g)
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
    se_i = se[iso]; si_i = si[iso]
    g = jnp.clip(1.0 + (se_i - 1.0) * (1.0 - finel) + (si_i - 1.0) * finel, 1e-6, None)
    a_nom, g = _match_dtype(a_nom, g)
    p0 = jnp.clip(jnp.exp(-a_nom), 1e-6, 1.0 - 1e-6)
    pk = jnp.clip(jnp.exp(-a_nom / g), 1e-6, 1.0 - 1e-6)
    s_real = jnp.where(inel, si_i, se_i)
    hit_f = (pk / p0) * (s_real / g)
    return jnp.where(hh, hit_f, (1.0 - pk) / (1.0 - p0))


def nncex_slot_factor(hh, iso, inel, swap, f_cex):
    """Per-slot NN-elastic charge-exchange FRACTION ratio (ACHILLES value 0.5).  Only pn ELASTIC hits carry
    a meaningful swap: f_cex/0.5 if swapped else (1-f_cex)/0.5.  pp/nn swaps are no-ops (same species)."""
    pn_el = hh & (~inel) & (iso == 1)
    return jnp.where(pn_el, jnp.where(swap, f_cex / 0.5, (1.0 - f_cex) / 0.5), 1.0)


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
    nucleus: str = "c12.prova.txt"
    density_n: str = "c12.prova.txt"
    configs: str = "QMC_configs.out.gz"
    step: float = 0.05
    max_steps: int = 100000
    path_budget_R: float = 20.0
    seed: int = 0
    time_step: bool = False
    fast_xsec: bool = True
    pauli: bool = True
    algo: str = "step"
    nn_inelastic: bool = True
    early_exit: bool = True
    track_steps: bool = False
    engine: str = "pool"
    recap_ke: float = 10.0


def sample_nucleons(key, n, cfg: CascadeConfig):
    """Pick n configurations ~ weight; assign each nucleon an isotropic local-Fermi-gas momentum.
    Returns npos (n,A,3), nmom (n,A,4), nisp (n,A) proton-mask."""
    pos, iso, w, A = _load_qmc_configs(name=cfg.configs)
    rgrid, rhoP, rhoN, _ = _load_density(cfg.nucleus, cfg.density_n)
    kc, kd, km = jax.random.split(key, 3)
    idx = jax.random.choice(kc, pos.shape[0], (n,), p=w)
    npos = pos[idx]; nisp = iso[idx]
    _ke = jax.random.fold_in(key, 777)
    _ang = jax.random.uniform(_ke, (n, 3)) * (2 * jnp.pi)
    _a, _b, _g = _ang[:, 0], _ang[:, 1] / 2.0, _ang[:, 2]
    c1, s1 = jnp.cos(_a), jnp.sin(_a); c2, s2 = jnp.cos(_b), jnp.sin(_b); c3, s3 = jnp.cos(_g), jnp.sin(_g)
    _R = jnp.stack([
        jnp.stack([c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2], axis=1),
        jnp.stack([c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2], axis=1),
        jnp.stack([s2 * s3, c3 * s2, c2], axis=1)], axis=1)
    npos = jnp.einsum('eij,eaj->eai', _R, npos)
    r = jnp.linalg.norm(npos, axis=2)
    kf = _kf_local(jnp.where(nisp, _rho_species(r, rgrid, rhoP), _rho_species(r, rgrid, rhoN)))
    d = jax.random.normal(kd, (n, A, 3)); d = d / jnp.linalg.norm(d, axis=2, keepdims=True)
    pm = kf * jax.random.uniform(km, (n, A)) ** (1.0 / 3.0)
    m_sp = jnp.where(nisp, _MP_PHYS, _MN_PHYS)
    p3 = d * pm[:, :, None]; E = jnp.sqrt(m_sp ** 2 + pm ** 2)
    nmom = jnp.concatenate([E[:, :, None], p3], axis=2)
    return npos, nmom, nisp


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
    """ONE step of the NUCLEON cascade for one particle per event (n,): escape/recapture, formation zone,
    in-slab geometry, elastic scatter + per-species Pauli, NN->NDelta->NN'pi inelastic + channel charges.
    Written for the POOLED engine: the knockout (recoil | inelastic 2nd nucleon) and the created pion are
    returned as immediate spawns (no top-K deferral), and the updated `consumed` mask is returned for
    slot-serialized depletion.  RNG usage (split(key,3) + fold_in(sk,101..108)) is fixed, so iterating
    this with the same per-step keys reproduces the leading trajectory bit-for-bit regardless of engine.
    Returns: (p4', pos', dhat', fz', alive'), terminal, recap, do, (ko4,kopos,kofz,koisp,koal),
             (pi4,pipos,pifz,pich,pial), consumed', (has_hit, perp2_c, sig_c)."""
    n, A = nisp.shape; ar = jnp.arange(n)
    esc_sphere = jnp.linalg.norm(pos, axis=1) > radius
    if is_beam is not None:
        reached = jnp.where(is_beam, pos[:, 2] >= radius, esc_sphere)
    else:
        reached = esc_sphere
    _e_phys = jnp.sqrt(jnp.where(isp, _MP_PHYS, _MN_PHYS) ** 2 + jnp.sum(p4[:, 1:] ** 2, axis=1))
    captured = reached & ((_e_phys - M_N) < cfg.recap_ke)
    escaping = reached & ~captured
    alive = alive & ~reached
    beta = jnp.linalg.norm(p4[:, 1:], axis=1) / jnp.clip(p4[:, 0], 1e-9, None)
    if cfg.time_step:
        _dt = jnp.full_like(beta, cfg.step) if dt_evt is None else dt_evt
        _dstep = beta * _dt
        timeStep = _dt
    else:
        _dstep = jnp.full_like(beta, cfg.step)
        timeStep = cfg.step / jnp.clip(beta, 1e-6, None)
    can_int = fz <= 0.0
    rel = npos - pos[:, None, :]
    par = jnp.sum(rel * dhat[:, None, :], axis=2)
    perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
    in_slab = (par > 0) & (par <= _dstep[:, None]) & (~consumed) & alive[:, None]
    Pp = p4[:, None, :] + nmom
    s = Pp[:, :, 0] ** 2 - jnp.sum(Pp[:, :, 1:] ** 2, axis=2)
    same_iso = isp[:, None] == nisp
    _m1_mev = jnp.where(isp, _MP_PHYS, _MN_PHYS)[:, None]
    _m2_mev = jnp.where(nisp, _MP_PHYS, _MN_PHYS)
    _m_pair_gev = 0.5 * (_m1_mev + _m2_mev) / 1000.0
    sqrts = jnp.sqrt(jnp.clip(s, (_m1_mev + _m2_mev) ** 2, None))
    sig_el = jnp.clip(nn_elastic_sigma(sqrts, same_iso, _m_pair_gev), 0.0, None)
    if cfg.nn_inelastic:
        _lam_in = (s - (_m1_mev + _m2_mev) ** 2) * (s - (_m1_mev - _m2_mev) ** 2)
        pcm = jnp.sqrt(jnp.clip(_lam_in, 0.0, None)) / (2.0 * sqrts) / 1000.0
        sig_in = jnp.clip(nni.sigma_nn_ndelta(sqrts / 1000.0, pcm, same_iso), 0.0, None)
    else:
        sig_in = jnp.zeros_like(sig_el)
    sig = sig_el + sig_in
    prob = jnp.where(in_slab, jnp.exp(-jnp.pi * perp2 / jnp.clip(sig * MB_TO_FM2, 1e-12, None)), 0.0)
    _ks3 = _ev_split(key, 3); sk, ku, ks = _ks3[:, 0], _ks3[:, 1], _ks3[:, 2]
    passes = in_slab & (_ev_uniform(ku, (A,)) < prob)
    big = jnp.where(passes, perp2, jnp.inf)
    j = jnp.argmin(big, axis=1)
    has_hit = jnp.isfinite(big[ar, j]) & alive & can_int
    perp2_is = jnp.where(in_slab, perp2, jnp.inf); cidx = jnp.argmin(perp2_is, axis=1)
    has_slab = jnp.any(in_slab, axis=1)
    perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6); sig_c = sig[ar, cidx]
    _nisp_c = nisp[ar, cidx]
    iso_c = jnp.where(isp & _nisp_c, 0, jnp.where((~isp) & (~_nisp_c), 2, 1)).astype(jnp.int32)
    finel_c = jnp.clip(sig_in[ar, cidx] / jnp.clip(sig_c, 1e-12, None), 0.0, 1.0)
    pN_j = nmom[ar, j]
    rnuc = jnp.linalg.norm(npos, axis=2)
    kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP), _rho_species(rnuc, rgrid, rhoN)))
    kf_j = kf_n[ar, j]
    _rnuc_j = rnuc[ar, j]
    kf_p_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoP))
    kf_n_j = _kf_local(_rho_species(_rnuc_j, rgrid, rhoN))
    _r_lead = jnp.linalg.norm(pos, axis=1)
    _kfp_l = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_l = _kf_local(_rho_species(_r_lead, rgrid, rhoN))

    _swap_cx = _ev_fold_uniform(sk, 109) < 0.5
    _struck_isp = nisp[ar, j]
    lead_isp_out = jnp.where(_swap_cx, _struck_isp, isp)
    rec_isp_out = jnp.where(_swap_cx, isp, _struck_isp)
    kf_lead = jnp.where(lead_isp_out, _kfp_l, _kfn_l)
    kf_rec = jnp.where(rec_isp_out, kf_p_j, kf_n_j)
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
    _mNb = jnp.where(isp, _MP_PHYS, _MN_PHYS)
    _mNs = jnp.where(nisp[ar, j], _MP_PHYS, _MN_PHYS)
    rs_j = jnp.sqrt(jnp.clip(Pj[:, 0] ** 2 - jnp.sum(Pj[:, 1:] ** 2, axis=1), (_mNb + _mNs) ** 2, None))
    u_m = _ev_fold_uniform(sk, 102)
    _m_d_raw = nni.sample_delta_mass(rs_j / 1000.0, u_m) * 1000.0
    cth1 = 2 * _ev_fold_uniform(sk, 103) - 1.0
    phi1 = 2 * jnp.pi * _ev_fold_uniform(sk, 104)
    _r2 = _ev_fold_uniform(sk, 105)
    _term2 = jnp.cbrt(9.0 - 18.0 * _r2 + 2.0 * jnp.sqrt(3.0)
                      * jnp.sqrt(jnp.clip(7.0 - 27.0 * _r2 + 27.0 * _r2 ** 2, 0.0, None)))
    cth2 = jnp.clip(1.0 / (jnp.cbrt(3.0) * _term2) - _term2 / jnp.cbrt(9.0), -1.0, 1.0)
    phi2 = 2 * jnp.pi * _ev_fold_uniform(sk, 106)
    q_pair = isp.astype(jnp.int32) + nisp[ar, j].astype(jnp.int32)
    u107 = _ev_fold_uniform(sk, 107)
    u108 = _ev_fold_uniform(sk, 108)
    dch = jnp.where(q_pair == 2, jnp.where(u107 < 0.75, 2, 1),
            jnp.where(q_pair == 1, jnp.where(u107 < 0.5, 1, 0),
                                   jnp.where(u107 < 0.25, 0, -1)))
    pi_q = jnp.where(dch == 2, 1,
            jnp.where(dch == 1, jnp.where(u108 < 1.0 / 3.0, 1, 0),
            jnp.where(dch == 0, jnp.where(u108 < 2.0 / 3.0, 0, -1), -1)))
    _mN1 = jnp.where((q_pair - dch) == 1, _MP_PHYS, _MN_PHYS)
    _mN2 = jnp.where((dch - pi_q) == 1, _MP_PHYS, _MN_PHYS)
    _mpi_dec = _CH_MASS[(1 - pi_q)]
    _m_d_hi = jnp.maximum(rs_j - _mN1, _MN_PHYS + _CH_MASS[0] + 1.0)
    m_d = jnp.clip(_m_d_raw, _MN_PHYS + _CH_MASS[0], _m_d_hi)

    def _split2(P4, mA, mB, cth_, phi_, aniso_axis=False):
        ss = jnp.clip(P4[:, 0] ** 2 - jnp.sum(P4[:, 1:] ** 2, axis=1), (mA + mB) ** 2 * 1.0001, None)
        rss = jnp.sqrt(ss)
        EA = (ss + mA ** 2 - mB ** 2) / (2 * rss)
        pf = jnp.sqrt(jnp.clip(EA ** 2 - mA ** 2, 0.0, None))
        sth_ = jnp.sqrt(jnp.clip(1 - cth_ ** 2, 0, None))
        if aniso_axis:
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

    pN1, pD = _split2(Pj, _mN1, m_d, cth1, phi1)
    pN2, _pPiX = _split2(pD, _mN2, _mpi_dec, cth2, phi2, aniso_axis=True)
    _kfp_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoP)); _kfn_lead = _kf_local(_rho_species(_r_lead, rgrid, rhoN))
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)
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
    bg_proton = rec_isp_out
    fz_new = _formation_zone(p4, p_out)
    nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)
    inel_nl = jnp.where(nl_is1[:, None], pN2, pN1)
    inel_nl_q = jnp.where(nl_is1, dch - pi_q, q_pair - dch)
    ko_cand = jnp.where(is_inel[:, None], inel_nl, recoil)
    ko_q = jnp.where(is_inel, inel_nl_q, bg_proton.astype(jnp.int32))
    lead_inel_q = jnp.where(nl_is1, q_pair - dch, dch - pi_q)
    lead_q_new = jnp.where(do, lead_isp_out.astype(jnp.int32),
                           jnp.where(is_inel, lead_inel_q, isp.astype(jnp.int32))).astype(jnp.int32)
    ko_fz = jnp.where(is_inel, _formation_zone(p4, inel_nl), _formation_zone(p4, recoil))
    ko_alive = (do | is_inel) & (jnp.linalg.norm(ko_cand[:, 1:], axis=1) > 1.0)
    ko_pos = npos[ar, j]
    _delta_pm = (dch == 0) | (dch == 1)
    _gamma = _delta_pm & (_ev_fold_uniform(sk, 111) < 0.0055)
    pi_alive = is_inel & (jnp.linalg.norm(_pPiX[:, 1:], axis=1) > 1.0) & ~_gamma
    pi_fz = _formation_zone(p4, _pPiX)
    pi_pos = jnp.broadcast_to(pos, _pPiX[:, 1:].shape)
    p4 = jnp.where(do[:, None], p_out, jnp.where(is_inel[:, None], lead_in, p4))
    consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & (do | is_inel)[:, None])
    fz = jnp.where((fz > 0.0) & alive, fz - timeStep, fz)
    fz = jnp.where(do, fz_new, jnp.where(is_inel, _formation_zone(p4, lead_in), fz))
    pos = pos + _dstep[:, None] * dhat * alive[:, None]
    d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
    return ((p4, pos, dhat, fz, alive, lead_q_new), escaping, captured, do.astype(jnp.int32),
            (ko_cand, ko_pos, ko_fz, ko_q, ko_alive),
            (_pPiX, pi_pos, pi_fz, pi_chidx, pi_alive), consumed,
            jax.lax.stop_gradient((has_hit, perp2_c, sig_c, iso_c, finel_c, is_inel, do & _swap_cx)))


def _pion_step(p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed,
               rgrid, rhoP, rhoN, radius, cfg, key, dt_evt=None, is_beam=None):
    """ONE step of the PION cascade for one pion per event (n,), algo="step" only.  Written for the
    POOLED engine: the absorption products (piNN->NN, up to 2 protons), the scatter recoil, and the
    eta-N' conversion baryon are returned as immediate NUCLEON spawns; the scattered pion continues in
    place (charge oscillates) and the absorbed/converted pion is removed.  RNG usage (split(sk,7) +
    fold_in(ka,211/212) + per-event splits) is fixed, so the same per-step keys reproduce the leading
    PION trajectory bit-for-bit regardless of engine.  ch = pion charge index (0=pi+,1=pi0,2=pi-).
    Returns: (p4', pos', dhat', ch', nsc', alive'), escaping, is_abs, is_conv,
             (s1_p4,s1_pos,s1_fz,s1_q,s1_al), (s2_p4,s2_pos,s2_fz,s2_q,s2_al), consumed', srec."""
    assert cfg.algo == "step", "pool pion step implements the 'step' algo (production path) only"
    n, A = nisp.shape; ar = jnp.arange(n)
    p_pi = p4
    is_eta = (ch == 3)
    ext = (nsc == 0) if is_beam is None else is_beam
    esc_sphere = jnp.linalg.norm(pos, axis=1) > radius
    escaping = jnp.where(ext, pos[:, 2] >= radius, esc_sphere)
    alive = alive & ~escaping
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
        ch_pi = jnp.clip(ch, 0, 2)
        Wf = Wl.reshape(-1); nuf = nuc_i.reshape(-1); chf = jnp.broadcast_to(ch_pi[:, None], (n, K_)).reshape(-1)
        sio = meson_baryon_xsec.jax_channel_sigmas_resolved(Wf, chf, nuf).reshape(n, K_, 3)
        ssl = jnp.clip(jnp.sum(sio, axis=-1), 0.0, None)
        sil = jnp.clip(meson_baryon_xsec.jax_conversion_sigma(Wf, chf, nuf).reshape(n, K_), 0.0, None)
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
    prob = jnp.where(cand, jnp.exp(-jnp.pi * perp2 / _sfm), 0.0)
    _ks7 = _ev_split(key, 7)
    sk, ku, kc, kf, ka, kab, knp = (_ks7[:, i] for i in range(7))
    passes = cand & (_ev_uniform(ku, (A,)) < prob)
    metric = jnp.where(passes, perp2, jnp.inf)
    j = jnp.argmin(metric, axis=1)
    has_hit = jnp.isfinite(metric[ar, j]) & alive
    sa_j = sa[ar, j]; si_j = si[ar, j]; sig_j = sig[ar, j]
    perp2_is = jnp.where(cand, perp2, jnp.inf)
    cidx = jnp.argmin(perp2_is, axis=1)
    has_slab = jnp.any(cand, axis=1)
    perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6)
    sa_c = sa[ar, cidx]; si_c = si[ar, cidx]
    ss_c = ss[ar, cidx]
    ss_el_c = sig_io.reshape(n, A, 3)[ar, cidx][ar, jnp.clip(ch, 0, 2)]
    W_j = W[ar, j]; pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
    kf_p_j = _kf_local(_rho_species(rnuc[ar, j], rgrid, rhoP))
    kf_n_j = _kf_local(_rho_species(rnuc[ar, j], rgrid, rhoN))
    p_abs = sa_j / jnp.clip(sig_j, 1e-12, None)
    p_conv = si_j / jnp.clip(sig_j, 1e-12, None)
    u_br = _ev_uniform(kc)
    chose_abs = has_hit & (u_br < p_abs)
    chose_conv = has_hit & ~chose_abs & (u_br < p_abs + p_conv)
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
        return pa, pb, A_is_p.astype(jnp.int32), B_is_p.astype(jnp.int32), blocked
    abs_pa, abs_pb, abs_qa, abs_qb, abs_blocked = jax.vmap(abs_one)(p_pi, pN_j, pN_p, nprot_out,
                                                         kfPA, kfNA, kfPB, kfNB, kab)
    if not cfg.pauli:
        abs_blocked = abs_blocked & False
    is_abs = chose_abs & ~abs_blocked & has_mode
    sig_io_j = sig_io.reshape(n, A, 3)[ar, j]
    probs = sig_io_j / jnp.clip(jnp.sum(sig_io_j, axis=1, keepdims=True), 1e-12, None)
    u = _ev_uniform(kf, (1,))
    out_ch = jnp.clip(jnp.sum((u > jnp.cumsum(probs, axis=1)).astype(jnp.int32), axis=1), 0, 2).astype(jnp.int32)
    out_ch = jnp.where(is_eta, jnp.int32(3), out_ch)
    nuc_idx = jnp.where(nisp[ar, j], 0, 1)
    chan_idx = jnp.clip(ch, 0, 2) * 6 + nuc_idx * 3 + jnp.clip(out_ch, 0, 2)
    u_ang = _ev_uniform(ka)
    cos_cm = jnp.where(is_eta, 2.0 * u_ang - 1.0,
                       meson_baryon_xsec.jax_sample_cos_cm(W_j, u_ang, chan_idx))
    _rec_is_p = jnp.where(is_eta, struck_p == 1, (struck_p + out_ch - ch) == 1)
    kf_rec_pi = jnp.where(_rec_is_p, kf_p_j, kf_n_j)
    m_rec_pi = jnp.where(_rec_is_p, _MP_PHYS, _MN_PHYS)

    def scat_one(p_pi_i, pN_i, out_i, kf_i, mrec, cc, k):
        p_out = _two_body_cm_scatter(p_pi_i, pN_i, _CH_MASS[out_i], k, cos_cm=cc, m_recoil=mrec)
        p_rec = (p_pi_i + pN_i) - p_out
        return p_out, jnp.linalg.norm(p_rec[1:]) < kf_i
    p_out, blocked = jax.vmap(scat_one)(p_pi, pN_j, out_ch, kf_rec_pi, m_rec_pi, cos_cm, sk)
    if not cfg.pauli:
        blocked = blocked & False
    is_conv = chose_conv
    nuc_idx_j = nuc_idx
    bc_j = meson_baryon_xsec.jax_eta_backconv_sigma(W_j, nuc_idx_j)
    pbc = bc_j / jnp.clip(jnp.sum(bc_j, axis=1, keepdims=True), 1e-12, None)
    u_out = _ev_fold_uniform(ka, 331)
    out_pi_idx = jnp.clip(jnp.sum((u_out[:, None] > jnp.cumsum(pbc, axis=1)).astype(jnp.int32), axis=1),
                          0, 2).astype(jnp.int32)
    si_eta_j = meson_baryon_xsec.jax_pi_to_eta_sigma(W_j, jnp.clip(ch, 0, 2), nuc_idx_j)
    frac_morph = jnp.where(is_eta, 1.0, si_eta_j / jnp.clip(si_j, 1e-12, None))
    chose_morph = is_conv & (_ev_fold_uniform(ka, 332) < frac_morph)
    m_out = jnp.where(is_eta, _CH_MASS[out_pi_idx], _CH_MASS[3])
    meson_q = jnp.where(is_eta, out_pi_idx, 3).astype(jnp.int32)
    q_bary = jnp.where(is_eta, struck_p - (1 - out_pi_idx), (1 - ch) + struck_p)
    morph_ok = chose_morph & ((q_bary == 0) | (q_bary == 1))
    _mB_conv = jnp.where(q_bary == 1, _MP_PHYS, _MN_PHYS)
    Pcv = p_pi + pN_j
    scv = Pcv[:, 0] ** 2 - jnp.sum(Pcv[:, 1:] ** 2, axis=1)
    rscv = jnp.sqrt(jnp.clip(scv, (_mB_conv + m_out) ** 2, None))
    EN = (scv + _mB_conv ** 2 - m_out ** 2) / (2.0 * rscv)
    Em = rscv - EN
    pst = jnp.sqrt(jnp.clip(EN ** 2 - _mB_conv ** 2, 0.0, None))
    ccv = 2.0 * _ev_fold_uniform(ka, 211) - 1.0
    scv_ = jnp.sqrt(jnp.clip(1 - ccv ** 2, 0.0, None)); phcv = 2 * jnp.pi * _ev_fold_uniform(ka, 212)
    dcv = jnp.stack([scv_ * jnp.cos(phcv), scv_ * jnp.sin(phcv), ccv], axis=1)
    beta = Pcv[:, 1:] / Pcv[:, [0]]; b2 = jnp.sum(beta ** 2, axis=1); gcv = 1 / jnp.sqrt(jnp.clip(1 - b2, 1e-12, None))

    def _boost_cm(Ecm, p3cm):
        bp = jnp.sum(beta * p3cm, axis=1)
        p3 = p3cm + ((gcv - 1) * bp / jnp.clip(b2, 1e-30, None) + gcv * Ecm)[:, None] * beta
        return jnp.concatenate([(gcv * (Ecm + bp))[:, None], p3], axis=1)
    p_bary = _boost_cm(EN, pst[:, None] * dcv)
    p_meson = _boost_cm(Em, -pst[:, None] * dcv)
    kf_bary = jnp.where(q_bary == 1, kf_p_j, kf_n_j)
    conv_blocked = chose_morph & (jnp.linalg.norm(p_bary[:, 1:], axis=1) < kf_bary)
    if not cfg.pauli:
        conv_blocked = conv_blocked & False
    morph_ok = morph_ok & ~conv_blocked
    is_conv = is_conv & ~conv_blocked
    is_scat = has_hit & ~chose_abs & ~chose_conv & ~blocked
    p_rec = (p_pi + pN_j) - p_out
    q_rec = struck_p + out_ch - ch
    rcand = jnp.where(is_conv[:, None], p_bary, p_rec)
    fz_rec = _formation_zone(p_pi, rcand)
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
    sm_p4 = jnp.where(morph_ok[:, None], p_meson, 0.0)
    sm_q = jnp.where(morph_ok, meson_q, 0).astype(jnp.int32)
    sm_pos = npos[ar, j]
    sm_fz = jnp.zeros_like(s1_fz)
    sm_al = morph_ok & (jnp.linalg.norm(sm_p4[:, 1:], axis=1) > 1.0)
    p_pi = jnp.where(is_scat[:, None], p_out, p_pi)
    ch = jnp.where(is_scat, out_ch, ch)
    nsc = nsc + is_scat.astype(jnp.int32)
    alive = alive & ~is_abs & ~is_conv
    interacted = is_abs | is_scat | is_conv
    consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & interacted[:, None])
    pos = pos + _dstep[:, None] * dhat * alive[:, None]
    d3 = p_pi[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
    ss_j = ss[ar, j]
    ss_el_j = sig_io_j[ar, jnp.clip(ch, 0, 2)]
    perp2_c = jnp.where(is_eta, 1e6, perp2_c)
    code4 = jnp.where(chose_abs, 2, jnp.where(chose_conv, 3,
                      jnp.where(out_ch == ch, 0, 1))).astype(jnp.int32)
    return ((p_pi, pos, dhat, ch, nsc, alive), escaping, is_abs, is_conv,
            (s1_p4, s1_pos, s1_fz, s1_q, s1_al), (s2_p4, s2_pos, s2_fz, s2_q, s2_al),
            (sm_p4, sm_pos, sm_fz, sm_q, sm_al), consumed,
            jax.lax.stop_gradient((has_hit, code4, sa_j, ss_el_j, ss_j, si_j,
                                   perp2_c, sa_c, ss_el_c, ss_c, si_c)))


PION, NUCLEON = 0, 1
FATE_NONE, FATE_ESCAPE, FATE_ABSORB, FATE_CONVERT, FATE_CAPTURE = 0, 1, 2, 3, 4
_ORIG_PRIM_PI = 2
_TRACK_OFFSET = 1000
_DEFAULT_NW = 2048
FLAT_FSI_REC = os.environ.get("ADONIS_FLAT_FSI", "0") != "0"
_DEFAULT_QCAP = 64
_HARD_STEPS = 100000


def _raise_if_runaway(counter, where):
    """Guard: raise if the cascade hit the _HARD_STEPS ceiling (a particle never terminated).
    Works eager and under jit: the tracer case is routed through jax.debug.callback so a runaway is
    never silently truncated (incl. on GPU)."""
    def _check(c):
        c = int(c)
        if c > 0:
            raise RuntimeError(
                f"cascade {where}: hit the {_HARD_STEPS}-step hard ceiling for {c} particle/event(s) -- "
                "no physical termination (escape/capture/absorption/path-budget) fired.  This should never "
                "happen; investigate (runaway particle) rather than accept a silently truncated cascade.")
    try:
        _check(counter)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        jax.debug.callback(_check, counter)


def setup_nucleus(p_pi, pid_pi, pid_Ni, cfg, key):
    """Sample the nucleus background + struck vertex (material from cfg), so the engine's primary-pion
    segment reproduces the production chain bit-for-bit.  Returns nucleus + pion init."""
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
                nsc=jnp.zeros((n, P), jnp.int32),
                external_test=jnp.zeros((n, P), bool),
                alive=jnp.zeros((n, P), bool), w=jnp.ones((n, P)), fate=jnp.zeros((n, P), jnp.int32),
                origin=jnp.zeros((n, P), jnp.int32), gen=jnp.zeros((n, P), jnp.int32),
                track_id=jnp.zeros((n, P), jnp.int32),
                parent_id=jnp.full((n, P), -1, jnp.int32),
                pkey=jnp.zeros((n, P, 2), jnp.uint32),
                lstep=jnp.zeros((n, P), jnp.int32),
                lpath=jnp.zeros((n, P)),
                gtime=jnp.zeros((n, P), jnp.int32))


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
        rank = jnp.cumsum(alive.astype(jnp.int32), axis=1) - 1
    else:
        order = jnp.lexsort((b["track_id"], b["gtime"], (~alive).astype(jnp.int32)), axis=-1)
        rank = jnp.argsort(order, axis=-1)
    rank = jnp.where(alive, rank, M + P_out)
    n_overflow = jnp.sum((alive & (rank >= P_out)).astype(jnp.int32))
    out = empty_batch(n, P_out)
    ar = jnp.broadcast_to(jnp.arange(n)[:, None], (n, M))
    dst = jnp.where(rank < P_out, rank, P_out)
    for k in ("species", "charge", "fate", "origin", "gen", "track_id", "parent_id", "nsc", "lstep", "gtime",
              "external_test"):
        if k not in b:
            continue
        out[k] = jnp.zeros((n, P_out + 1), b[k].dtype).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    for k in ("fz", "w", "lpath"):
        out[k] = jnp.zeros((n, P_out + 1)).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    out["alive"] = jnp.zeros((n, P_out + 1), bool).at[ar, dst].set(b["alive"], mode="drop")[:, :P_out]
    out["p4"] = jnp.zeros((n, P_out + 1, 4)).at[ar, dst].set(b["p4"], mode="drop")[:, :P_out]
    out["pos"] = jnp.zeros((n, P_out + 1, 3)).at[ar, dst].set(b["pos"], mode="drop")[:, :P_out]
    if "pkey" in b:
        out["pkey"] = jnp.zeros((n, P_out + 1, 2), b["pkey"].dtype).at[ar, dst].set(b["pkey"], mode="drop")[:, :P_out]
    return out, n_overflow


def make_pool_stepper(su, cfg, with_rec=False, with_seg=False):
    """Build the pooled-engine physics stepper: advance every slot of the (n, M) stack one step, dispatched
    by species (NUCLEON -> _nucleon_step, PION -> _pion_step), with the consumed mask threaded slot-serially
    (slot m+1 sees m's depletion).  Each slot emits up to 2 NUCLEON spawns + 1 PION spawn.  Returns
    stepper(stack, key, consumed) -> (stack2, terminal (n,M) bool, spawn ParticleBatch (n,3M), consumed)."""
    npos0, nmom0, nisp0 = su["npos"], su["nmom"], su["nisp"]
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    _dead = lambda n: (jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n, bool))

    def stepper(stack, key, consumed, step=0, bg=None, dt_evt=None):
        npos, nmom, nisp = (npos0, nmom0, nisp0) if bg is None else bg
        n, M = stack["alive"].shape
        _dt_e = dt_evt if dt_evt is not None else jnp.full(n, cfg.step)

        def slot(consumed, m):
            p4 = stack["p4"][:, m]; pos = stack["pos"][:, m]; fz = stack["fz"][:, m]
            nsc = stack["nsc"][:, m]; chg = stack["charge"][:, m]; al = stack["alive"][:, m]
            sp = stack["species"][:, m]
            is_N = (sp == NUCLEON) & al; is_pi = (sp == PION) & al
            d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
            pkey_m = stack["pkey"][:, m]; lstep_m = stack["lstep"][:, m]
            step_key = jax.vmap(lambda pk, ls: jax.random.fold_in(pk, ls))(pkey_m, lstep_m)
            _kk = jax.vmap(lambda k: jax.random.split(k))(step_key)
            kN, kP = _kk[:, 0], _kk[:, 1]
            (p4n, posn, _dn, fzn, alnN, qln), escN, capturedN, _do, koN, pinN, consumedN, nstat = _nucleon_step(
                p4, pos, dhat, fz, chg.astype(bool), is_N, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kN, dt_evt=_dt_e,
                is_beam=stack["external_test"][:, m])
            (p4p, posp, _dp, chp, nscp, alnP), escP, is_abs, is_conv, s1, s2, smes, consumedP, pstat = _pion_step(
                p4, pos, dhat, chg, nsc, is_pi, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kP, dt_evt=_dt_e,
                is_beam=stack["external_test"][:, m])
            if with_rec:
                (p_hh, p_bc, p_sa, p_ss_el, p_ss, p_si,
                 p_perp2_c, p_sa_c, p_ss_el_c, p_ss_c, p_si_c) = pstat
                n_hh, n_perp2, n_sig, n_iso, n_finel, n_inel, n_swap = nstat
                a_nom = jnp.pi * n_perp2 / jnp.clip(n_sig * MB_TO_FM2, 1e-12, None)
                p_sig_c = jnp.clip(p_sa_c + p_ss_c + p_si_c, 1e-12, None)
                p_a_nom = jnp.pi * p_perp2_c / jnp.clip(p_sig_c * MB_TO_FM2, 1e-12, None)
                p_fin = jnp.isfinite(p_a_nom) & jnp.isfinite(p_sa_c) & jnp.isfinite(p_ss_c) \
                    & jnp.isfinite(p_si_c) & jnp.isfinite(p_ss_el_c)
                rec_slot = (is_pi & (p_perp2_c < 1e5) & p_fin, p_bc, p_sa, p_ss_el, p_ss, p_si,
                            p_hh, p_a_nom, p_sa_c, p_ss_el_c, p_ss_c, p_si_c,
                            is_N & (n_perp2 < 1e5), n_hh, a_nom, n_iso, n_finel, n_inel, n_swap)
            consumed = consumedN | consumedP
            p4_2 = jnp.where(is_N[:, None], p4n, jnp.where(is_pi[:, None], p4p, p4))
            pos_2 = jnp.where(is_N[:, None], posn, jnp.where(is_pi[:, None], posp, pos))
            fz_2 = jnp.where(is_N, fzn, fz)
            n_react = is_N & (_do.astype(bool) | nstat[5])
            nsc_2 = jnp.where(is_pi, nscp, nsc + n_react.astype(jnp.int32))
            _react = n_react | (is_pi & ((nscp > nsc) | is_abs | is_conv))
            et_2 = stack["external_test"][:, m] & ~_react
            chg_2 = jnp.where(is_N, qln, jnp.where(is_pi, chp, chg))
            al_2 = jnp.where(is_N, alnN, jnp.where(is_pi, alnP, al))
            _beta_m = jnp.linalg.norm(d3, axis=1) / jnp.clip(p4[:, 0], 1e-9, None)
            _dstep_m = _beta_m * _dt_e if cfg.time_step else jnp.full_like(_beta_m, cfg.step)
            _lpath_2 = stack["lpath"][:, m] + _dstep_m * al.astype(stack["lpath"].dtype)
            al_2 = al_2 & (_lpath_2 < cfg.path_budget_R * radius) & ((lstep_m + 1) < cfg.max_steps)
            term = (is_N & escN) | (is_pi & escP)
            fate_2 = jnp.where(is_N & capturedN, FATE_CAPTURE,
                     jnp.where((is_N & escN) | (is_pi & escP), FATE_ESCAPE,
                     jnp.where(is_pi & is_abs, FATE_ABSORB,
                     jnp.where(is_pi & is_conv, FATE_CONVERT, FATE_NONE)))).astype(jnp.int32)
            new = (p4_2, pos_2, fz_2, nsc_2, chg_2, al_2, term, fate_2, et_2)
            nuc1 = tuple(jnp.where(is_N, kn, jnp.where(is_pi, s1k, dk)) if kn.ndim == 1
                         else jnp.where(is_N[:, None], kn, jnp.where(is_pi[:, None], s1k, dk))
                         for kn, s1k, dk in zip(koN, s1, _dead(n)))
            nuc2 = tuple(jnp.where(is_pi, s2k, dk) if s2k.ndim == 1
                         else jnp.where(is_pi[:, None], s2k, dk)
                         for s2k, dk in zip(s2, _dead(n)))
            pio = tuple(jnp.where(is_N, pk, jnp.where(is_pi, mk, dk)) if pk.ndim == 1
                        else jnp.where(is_N[:, None], pk, jnp.where(is_pi[:, None], mk, dk))
                        for pk, mk, dk in zip(pinN, smes, _dead(n)))
            seg_slot = None
            if with_seg:
                _PIDPI = jnp.array([211, 111, -211])
                pi_scat = is_pi & (nscp > nsc)
                made_pi = is_N & pinN[4]
                chan = jnp.where(made_pi, 2,
                        jnp.where(is_N & (_do > 0), 1,
                        jnp.where(is_pi & is_abs, 3,
                        jnp.where(is_pi & is_conv, 4,
                        jnp.where(pi_scat & (chp == chg), 1,
                        jnp.where(pi_scat & (chp != chg), 2, 0)))))).astype(jnp.int32)
                incp = jnp.linalg.norm(p4[:, 1:], axis=1)
                inc_pid = jnp.where(sp == NUCLEON, jnp.where(chg == 1, 2212, 2112),
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
        T = lambda x: jnp.moveaxis(x, 0, 1)
        p4s, poss, fzs, nscs, chgs, als, terms, fates, ets = new
        _beta_all = jnp.linalg.norm(stack["p4"][:, :, 1:], axis=2) / jnp.clip(stack["p4"][:, :, 0], 1e-9, None)
        _dstep_all = _beta_all * _dt_e[:, None] if cfg.time_step else jnp.full_like(_beta_all, cfg.step)
        stack2 = {**stack, "p4": T(p4s), "pos": T(poss), "fz": T(fzs), "nsc": T(nscs),
                  "charge": T(chgs), "alive": T(als), "fate": T(fates), "external_test": T(ets),
                  "lstep": stack["lstep"] + stack["alive"].astype(jnp.int32),
                  "lpath": stack["lpath"] + _dstep_all * stack["alive"].astype(stack["lpath"].dtype),
                  "gtime": stack["gtime"] + stack["alive"].astype(jnp.int32)}
        terminal = T(terms)
        if with_rec:
            rk = ("pi_m", "pi_bc", "pi_sa", "pi_ss_el", "pi_ss", "pi_si",
                  "pi_hh", "pi_a", "pi_sa_c", "pi_ss_el_c", "pi_ss_c", "pi_si_c",
                  "nu_m", "nu_hh", "nu_a", "nu_iso", "nu_finel", "nu_inel", "nu_swap")
            rec = {k: T(v) for k, v in zip(rk, scanned[4])}
        if with_seg:
            sk = ("chan", "incp", "inc_pid", "parent_id", "gen", "track_id", "cont_pid", "cont_p",
                  "n1_pid", "n1_p", "n1_al", "n2_pid", "n2_p", "n2_al", "pio_pid", "pio_p", "pio_al")
            seg = {k: T(v) for k, v in zip(sk, scanned[4 + (1 if with_rec else 0)])}

        def _spawn(species, packs):
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
        _pslot = jnp.tile(jnp.arange(M), 3)
        _chan = jnp.repeat(jnp.arange(3), M)
        _ppk = stack["pkey"][:, _pslot]
        _pls = stack["lstep"][:, _pslot]
        _dk = lambda pk, c, ls: jax.random.fold_in(jax.random.fold_in(pk, c), ls)
        spawn["pkey"] = jax.vmap(jax.vmap(_dk))(_ppk, jnp.broadcast_to(_chan, (n, 3 * M)), _pls)
        spawn["lstep"] = jnp.zeros((n, 3 * M), jnp.int32)
        spawn["gtime"] = stack["gtime"][:, _pslot] + 1
        sa = jnp.asarray(step)
        if sa.ndim == 0:
            tid = (_TRACK_OFFSET + (step * M + _pslot) * 3 + _chan).astype(jnp.int32)
            spawn["track_id"] = jnp.broadcast_to(tid[None, :], (n, 3 * M))
        else:
            tid = (_TRACK_OFFSET + (sa[:, None] * M + _pslot[None, :]) * 3 + _chan[None, :]).astype(jnp.int32)
            spawn["track_id"] = tid
        if with_seg:
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
    """The pooled engine's per-step in/out: drop the slots that terminated this step, keep the survivors,
    insert the particles created this step, and re-pack to width M (compact(concat(survivors, spawned), M)).
      stack/terminal/spawn/M : the post-step stack, its per-slot terminal flags, this step's spawns, width
      wait, Q : FIFO waiting buffer (n, Q) and its width (0 = no queue, overflow dropped)
    With Q>0 survivors+wait+spawn are packed into M active + Q waiting (survivors keep their slots, then
    the waiting buffer re-enters before new spawns); only particles beyond M+Q are dropped.
    Returns (new_stack (n, M), new_wait (n, Q) or None, overflow (scalar))."""
    stack = {**stack, "alive": stack["alive"] & ~terminal}
    if Q > 0 and wait is not None:
        combined = {k: jnp.concatenate([stack[k], wait[k], spawn[k]], axis=1) for k in stack}
        full, ndrop = compact(combined, M + Q, sort_priority=True)
        active = {k: v[:, :M] for k, v in full.items()}
        new_wait = {k: v[:, M:M + Q] for k, v in full.items()}
        return active, new_wait, ndrop
    combined = {k: jnp.concatenate([stack[k], spawn[k]], axis=1) for k in stack}
    active, ndrop = compact(combined, M, sort_priority=True)
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
    """Flat/streaming twin of _rec_scatter: append this step's masked slots' `vals` into ONE flat (cap,)
    buffer at a global running cursor `gc`, tagging each written slot with its event id in `eidx_buf`.
    `cap` = total interaction budget (tail-insensitive), not per-event K.  Overflow is dropped.  Unwritten
    eidx slots keep n_events (out of range) so _ragged_prod drops them.  The reweight is a scatter-add by
    eidx, so step-order here gives a bit-identical reweight to _rec_scatter's per-event order.
    Returns (new_bufs, new_gc, new_eidx_buf, overflow_this_step)."""
    n, M = mask.shape
    fm = mask.reshape(-1)
    order = jnp.cumsum(fm.astype(jnp.int32)) - 1
    gpos = gc + order
    valid = fm & (gpos < cap)
    dst = jnp.where(valid, gpos, cap)
    eidx_flat = jnp.repeat(evt_id, M)
    new_bufs = [b.at[dst].set(v.reshape(-1), mode="drop") for b, v in zip(bufs, vals)]
    new_eidx = eidx_buf.at[dst].set(jnp.where(valid, eidx_flat, n_events), mode="drop")
    overflow = jnp.sum((fm & (gpos >= cap)).astype(jnp.int32)).astype(jnp.int32)
    return new_bufs, (gc + jnp.sum(fm.astype(jnp.int32))).astype(gc.dtype), new_eidx, overflow


def _empty_flat_fsi_record(Tp, Tn, n_events):
    """Flat FSI record: single (Tp,)/(Tn,) field buffers + global cursors gc_p/gc_n + per-slot event index
    p_eidx/n_eidx (init n_events, out of range -> unwritten slots dropped by _ragged_prod).  Slot defaults
    match _empty_fsi_record (per-slot LR=1 at nominal)."""
    return dict(bc=jnp.zeros(Tp, jnp.int32), sa=jnp.ones(Tp), ss_el=jnp.ones(Tp), ss=jnp.ones(Tp),
                si=jnp.zeros(Tp), pi_hh=jnp.zeros(Tp, bool), pi_a=jnp.full(Tp, 50.0),
                sa_c=jnp.ones(Tp), ss_el_c=jnp.ones(Tp), ss_c=jnp.ones(Tp), si_c=jnp.zeros(Tp),
                hh=jnp.zeros(Tn, bool), a=jnp.full(Tn, 50.0), iso=jnp.zeros(Tn, jnp.int32),
                finel=jnp.zeros(Tn), inel=jnp.zeros(Tn, bool), swap=jnp.zeros(Tn, bool),
                gc_p=jnp.int32(0), gc_n=jnp.int32(0),
                p_eidx=jnp.full(Tp, n_events, jnp.int32), n_eidx=jnp.full(Tn, n_events, jnp.int32))


def _empty_fsi_record(n, Kp, Kn):
    """Per-event kind-1 FSI reweight buffers (defaults give per-slot LR=1).  bc is the granular pion channel
    code {0 el,1 cex,2 abs,3 conv}; ss_el = elastic part of ss (ss_cex = ss-ss_el).  Pion slots are per
    in-slab candidate step (hit or not): pi_hh/pi_a/*_c carry the sigma_tot response, without which a
    common rescale of the four pion sigmas would be a spurious flat direction."""
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
    """Joint kind-1 FSI reweight for the pool.  Pion branch-split + nucleon (per-iso el/inel).  Pure in
    the scales; == 1 at nominal; == the in-walk weight at any theta.  pool_fsi_reweight(record, sabs,
    sscat) reproduces the coarse two-knob reweight bit-for-bit (granular knobs default to sabs/sscat)."""
    s_el = sscat if s_piN_elastic is None else s_piN_elastic
    s_cex = sscat if s_piN_cex is None else s_piN_cex
    if "p_eidx" in record:
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
                            f_NN_cex)
    return wp * wn * wc


_P_SLOT = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c")
_N_SLOT = ("hh", "a", "iso", "finel", "inel", "swap")


def compact_fsi_record(rec):
    """Dense (n, K) kind-1 record -> ragged: flat slot arrays + per-slot event index (p_eidx / n_eidx).
    Pure numpy, called once at bank-write time; only drops the padding.  A flat record (carries gc_p/gc_n)
    is already ragged -- just trim to the cursor."""
    import numpy as _np
    if "gc_p" in rec:
        gp = int(_np.asarray(rec["gc_p"])); gn = int(_np.asarray(rec["gc_n"]))
        Tp = len(_np.asarray(rec["bc"])); Tn = len(_np.asarray(rec["hh"]))
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
        if cmax > K:
            raise ValueError(
                f"FSI '{tag}' record overflow: max per-event steps = {cmax} exceeds the buffer cap K = {K} "
                f"(rec_caps). Raise rec_caps for this species above {cmax}, or ADONIS_REC_MARGIN, which "
                f"scales the caps the bank generator calibrates.")
        keep = _np.arange(K)[None, :] < c[:, None]
        out[f"{tag}_eidx"] = _np.repeat(_np.arange(len(c), dtype=_np.int32), c)
        for f in names:
            out[f] = _np.asarray(rec[f])[keep]
        assert len(out[f"{tag}_eidx"]) == int(keep.sum())
    return out


def pool_fsi_reweight_flat(record, sabs, sscat, *, s_piN_elastic=None, s_piN_cex=None, s_conv=1.0,
                           s_NN_elastic=None, s_NN_inelastic=None, f_NN_cex=0.5):
    """pool_fsi_reweight on a RAGGED record (compact_fsi_record output + n_events).  Same per-slot physics
    (pion_slot_factor / nucleon_slot_factor / nncex_slot_factor), ragged reduction.  record["n_events"]
    gives the event count."""
    R = record
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


_LOG_I = ("chan", "inc_pid", "parent_id", "gen", "track_id", "cont_pid", "n1_pid", "n1_al",
          "n2_pid", "n2_al", "pio_pid", "pio_al")
_LOG_F = ("incp", "cont_p", "n1_p", "n2_p", "pio_p")


def _take_rows(d, idx):
    """Gather rows (leading axis) of a pytree-of-arrays dict at integer index `idx`."""
    return {k: v[idx] for k, v in d.items()}


def _round_betamax(stk, wait, round_gt, betamax):
    """ACHILLES AdaptiveStep, P-invariantly: per-event beta_max = max beta over all the event's alive
    particles (stack union wait, a P-independent set), snapshotted once per gtime-round (detected when the
    event's min(gtime) over alive particles advances past `round_gt`).  Within a round the stored beta_max
    is reused.  Returns (round_gt', beta_max').  dt_evt = step/beta_max, so a particle alone moves a full
    step (no freeze, no arbitrary cap)."""
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
    """POOLED engine loop: ONE fixed-size (W, M) particle stack stepped once per step; the in/out
    reconcile (pool_reconcile) runs inside the step.
      stepper(stack, key, state) -> (stack2 (W,M), terminal (W,M) bool, spawn (W,K), state2[, rec|seg])
    Escaped terminals accumulate into a fixed (W, M_out) output batch.  RNG is per event (key folded by
    evt_id+nstep) so an event is reproducible regardless of slot/step.

    Two modes share the per-step body `_apply_step`:
      * NO-REFILL (pending=None, default): the working set IS the n events, run lock-step until done.
      * REFILL (pending given): the working set holds `n_w` event-slots fed from a pending pool of all
        N_total events; when a slot's event finishes or hits its per-event step cap (`per_event_cap`,
        replacing the global max_steps), its accumulators flush to global buffers at its evt_id and the
        slot refills from a cursor.  At n_w=N_total it is bit-exact to no-refill; at n_w<N_total the
        per-event outputs are identical (compare by evt_id), only occupancy/wall time changes.
    `prim_origin` (RES): origin tag of the PRIMARY pion -> its terminal fate is latched per event.
    rec_caps=(Kp,Kn): accumulate the per-event kind-1 FSI reweight record.  log_cap=L: in-engine segment
    logger.  rec_caps and log_cap are mutually exclusive.
    Returns (out_batch, stack_overflow, out_overflow, prim_fate[, fsi_record | (log, counts, log_overflow)])."""
    with_rec = rec_caps is not None
    do_log = log_cap is not None
    Wprim = init["species"].shape[1]
    assert not (with_rec and do_log), "rec_caps and log_cap are mutually exclusive"
    Kp, Kn = rec_caps if with_rec else (1, 1)
    L = int(log_cap) if do_log else 1
    Q = int(q_cap) if q_cap and q_cap > 0 else 0
    cap = int(per_event_cap) if per_event_cap else int(max_steps)

    def _logbuf(W):
        return {**{k: jnp.zeros((W, L), jnp.int32) for k in _LOG_I},
                **{k: jnp.zeros((W, L), jnp.float32) for k in _LOG_F}}

    def _apply_step(stk, state, kk, step_i, wait, out, prim, rb, log, wptr, sofl, oofl, rofl, logofl, bg_w, ar,
                    round_gt, betamax, evt_id=None):
        """ONE pooled step on the working set + accumulation (out, prim latch, rec, seg log) + reconcile.
        Scalars sofl/oofl/rofl/logofl accumulate globally; out/prim/rb/log/wptr are per-slot.  Identical
        maths regardless of refill mode.  time_sync: ACHILLES AdaptiveStep per-event dt = step/beta_max."""
        if time_sync:
            round_gt, betamax = _round_betamax(stk, wait, round_gt, betamax)
            dt_evt = jnp.full_like(betamax, step) if os.environ.get("ADONIS_DECOUPLED") == "1" else step / betamax
        else:
            dt_evt = None
        if bg_w is not None:
            _step = stepper(stk, kk, state, step_i, bg=bg_w, dt_evt=dt_evt)
        else:
            _step = stepper(stk, kk, state, step_i, dt_evt=dt_evt)
        stk2, terminal, spawn, state2 = _step[:4]
        extra = _step[4] if len(_step) >= 5 else None
        term_batch = {**stk2, "alive": terminal}
        out2, oo = compact({k: jnp.concatenate([out[k], term_batch[k]], axis=1) for k in out}, M_out)
        isprim = (stk2["track_id"] < _TRACK_OFFSET) & (stk2["fate"] != FATE_NONE)
        for _w in range(prim.shape[1]):
            _m = isprim & (stk2["track_id"] == _w)
            _any = jnp.any(_m, axis=1); _j = jnp.argmax(_m, axis=1)
            prim = prim.at[:, _w].set(jnp.where(_any & (prim[:, _w] == FATE_NONE), stk2["fate"][ar, _j], prim[:, _w]))
        if with_rec:
            rec = extra
            pion_vals = [rec["pi_bc"], rec["pi_sa"], rec["pi_ss_el"], rec["pi_ss"], rec["pi_si"],
                         rec["pi_hh"], rec["pi_a"], rec["pi_sa_c"], rec["pi_ss_el_c"], rec["pi_ss_c"],
                         rec["pi_si_c"]]
            nuc_vals = [rec["nu_hh"], rec["nu_a"], rec["nu_iso"], rec["nu_finel"], rec["nu_inel"],
                        rec["nu_swap"]]
            if "gc_p" in rb:
                new_p, gc_p, p_eidx, op = _rec_scatter_flat(
                    [rb[f] for f in _P_SLOT], rb["gc_p"], rb["p_eidx"], rec["pi_m"], pion_vals, evt_id, Kp, _NEVT)
                new_n, gc_n, n_eidx, on = _rec_scatter_flat(
                    [rb[f] for f in _N_SLOT], rb["gc_n"], rb["n_eidx"], rec["nu_m"], nuc_vals, evt_id, Kn, _NEVT)
                rb = {**dict(zip(_P_SLOT, new_p)), **dict(zip(_N_SLOT, new_n)),
                      "gc_p": gc_p, "gc_n": gc_n, "p_eidx": p_eidx, "n_eidx": n_eidx}
            else:
                new_p, nh, op = _rec_scatter([rb[f] for f in _P_SLOT], rb["nh"], rec["pi_m"], pion_vals, Kp)
                new_n, ns, on = _rec_scatter([rb[f] for f in _N_SLOT], rb["ns"], rec["nu_m"], nuc_vals, Kn)
                rb = {**dict(zip(_P_SLOT, new_p)), **dict(zip(_N_SLOT, new_n)), "nh": nh, "ns": ns}
            rofl = (rofl + op + on).astype(rofl.dtype)
        if do_log:
            seg = extra
            ev = (seg["chan"] > 0) | terminal
            evi = ev.astype(jnp.int32)
            off = jnp.cumsum(evi, axis=1) - evi
            tgt0 = wptr[:, None] + off
            logofl = logofl + (ev & (tgt0 >= L)).sum().astype(logofl.dtype)
            tgt = jnp.where(ev & (tgt0 < L), tgt0, L)
            log = dict(log)
            for k in _LOG_I + _LOG_F:
                log[k] = log[k].at[ar[:, None], tgt].set(seg[k].astype(log[k].dtype), mode="drop")
            wptr = wptr + evi.sum(axis=1).astype(wptr.dtype)
        if Q > 0:
            newstk, newwait, so = pool_reconcile(stk2, terminal, spawn, M, wait, Q)
        else:
            newstk, _nw, so = pool_reconcile(stk2, terminal, spawn, M)
            newwait = wait
        sofl = sofl + so.astype(sofl.dtype); oofl = oofl + oo.astype(oofl.dtype)
        return newstk, state2, newwait, out2, prim, rb, log, wptr, sofl, oofl, rofl, logofl, round_gt, betamax

    if pending is None:
        n = init["alive"].shape[0]; ar = jnp.arange(n)
        _NEVT = n
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
                   jnp.full((n, Wprim), FATE_NONE, jnp.int32), rec0, jnp.int32(0), log0, wptr0, logofl0, wait0,
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

    Ntot = pending["stack"]["alive"].shape[0]
    W = min(int(n_w) if n_w else Ntot, Ntot); ar = jnp.arange(W)
    _NEVT = Ntot
    _flat = with_rec and FLAT_FSI_REC
    if Q > 0:
        _pfull, _ = compact(pending["stack"], M + Q, sort_priority=True)
        pstack = {k: v[:, :M] for k, v in _pfull.items()}
        pwait = {k: v[:, M:M + Q] for k, v in _pfull.items()}
    else:
        pstack, _ = compact(pending["stack"], M, sort_priority=True)
        pwait = empty_batch(Ntot, 1)
    pbg = (pending["npos"], pending["nmom"], pending["nisp"])
    pcons = pending["consumed0"]
    g_out = empty_batch(Ntot, M_out); g_prim = jnp.full((Ntot, Wprim), FATE_NONE, jnp.int32)
    g_rb = {} if _flat else _empty_fsi_record(Ntot, Kp, Kn)
    g_log = _logbuf(Ntot); g_wptr = jnp.zeros(Ntot, jnp.int32)
    idx0 = jnp.arange(W, dtype=jnp.int32)
    stk0 = _take_rows(pstack, idx0); cons0 = pcons[idx0]
    bg0 = (pbg[0][idx0], pbg[1][idx0], pbg[2][idx0])
    out0 = empty_batch(W, M_out)
    rb0 = _empty_flat_fsi_record(Kp, Kn, Ntot) if _flat else _empty_fsi_record(W, Kp, Kn)
    log0 = _logbuf(W); wptr0 = jnp.zeros(W, jnp.int32)
    prim0 = jnp.full((W, Wprim), FATE_NONE, jnp.int32)
    wait0 = _take_rows(pwait, idx0) if Q > 0 else empty_batch(W, max(Q, 1))
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
        hit_cap = (nstep2 >= cap)
        forced = (forced + jnp.sum((hit_cap & still_alive & (evt >= 0)).astype(jnp.int32))).astype(jnp.int32)
        finished = (~still_alive) | hit_cap
        flush = finished & (evt >= 0)
        gi = jnp.where(flush, evt, Ntot)
        go = {k: go[k].at[gi].set(out2[k], mode="drop") for k in go}
        gp = gp.at[gi].set(prim2, mode="drop")
        if not _flat:
            grb = {k: grb[k].at[gi].set(rb2[k], mode="drop") for k in grb}
        glog = {k: glog[k].at[gi].set(log2[k], mode="drop") for k in glog}
        gwp = gwp.at[gi].set(wptr2, mode="drop")
        rank = jnp.cumsum(flush.astype(jnp.int32)) - flush.astype(jnp.int32)
        new_id = (cursor + rank).astype(jnp.int32)
        take = flush & (new_id < Ntot)
        cursor2 = (cursor + jnp.sum(take.astype(jnp.int32))).astype(jnp.int32)
        gidx = jnp.where(take, new_id, 0)
        rstk = _take_rows(pstack, gidx); rbg = (pbg[0][gidx], pbg[1][gidx], pbg[2][gidx]); rcons = pcons[gidx]
        rwait = _take_rows(pwait, gidx) if Q > 0 else empty_batch(W, max(Q, 1))
        def _sel(new, old, mask, md=None):
            m = mask.reshape((-1,) + (1,) * (old.ndim - 1))
            return jnp.where(m, new, old)
        idle = finished & (~take)
        stk3 = {k: _sel(rstk[k], newstk[k], take) for k in newstk}
        stk3["alive"] = jnp.where(idle[:, None], False, stk3["alive"])
        bg3 = tuple(_sel(rbg[t], bgw[t], take) for t in range(3))
        cons3 = _sel(rcons, cons2, take)
        _ewait = empty_batch(W, max(Q, 1))
        wait3 = {k: _sel(rwait[k], _sel(_ewait[k], newwait[k], finished), take) for k in newwait}
        e_out = empty_batch(W, M_out); e_log = _logbuf(W)
        out3 = {k: _sel(e_out[k], out2[k], finished) for k in out2}
        if _flat:
            rb3 = rb2
        else:
            e_rb = _empty_fsi_record(W, Kp, Kn)
            rb3 = {k: _sel(e_rb[k], rb2[k], finished) for k in rb2}
        log3 = {k: _sel(e_log[k], log2[k], finished) for k in log2}
        prim3 = jnp.where(finished[:, None], FATE_NONE, prim2)
        wptr3 = jnp.where(finished, 0, wptr2)
        evt3 = jnp.where(finished, jnp.where(take, new_id, jnp.int32(-1)), evt)
        nstep3 = jnp.where(finished, jnp.int32(0), nstep2)
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
    if with_rec:
        return go, sofl, oofl, gp, (_rb_fin if _flat else grb, rofl)
    return go, sofl, oofl, gp


def _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, rec_caps=None, log_cap=None,
                  n_w=0, q_cap=0, per_event_cap=None):
    """POOLED-engine realization of cascade_nucleus (QE + RES), mapping the flat (n,M_out) terminal
    buffer back to the rich (pterm, nterms, overflow, created) schema.
    log_cap=L: run the same cascade with the in-engine segment logger on and return (log, counts,
    log_overflow) directly -- the single entry point for the cascade-vertex/segment matrix.
      QE : gen-0 stack = the struck->proton (1 NUCLEON slot); pterm = QE "none"; created = leading
           surviving pion.
      RES: gen-0 stack = the PRIMARY pion (origin-tagged) + the RES recoil nucleon; the pion's own
           scatter-recoils / absorption protons / created pions spawn natively during the walk.  pterm =
           the primary pion's outcome (escape/absorbed/converted, via the latched fate); created = leading
           surviving non-primary pion."""
    ar = jnp.arange(n)
    if channel == "qe":
        g0 = empty_batch(n, 1)
        g0["alive"] = jnp.ones((n, 1), bool)
        g0["species"] = jnp.full((n, 1), NUCLEON, jnp.int32)
        g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]
        g0["p4"] = p_N[:, None, :]; g0["pos"] = su["pos0"][:, None, :]
        g0["track_id"] = jnp.zeros((n, 1), jnp.int32)
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))
        g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]
        prim_origin = -999
    else:
        g0 = empty_batch(n, 2)
        g0["alive"] = jnp.ones((n, 2), bool)
        g0["species"] = jnp.array([PION, NUCLEON], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
        g0["p4"] = jnp.stack([p_pi, p_N], axis=1)
        g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (n, 2, 3))
        g0["origin"] = jnp.array([_ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        g0["track_id"] = jnp.array([0, 1], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))
        g0["pkey"] = jnp.stack([jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base),
                                jax.vmap(lambda b: jax.random.fold_in(b, 1))(_base)], axis=1)
        prim_origin = _ORIG_PRIM_PI
    pend = dict(stack=g0, consumed0=su["consumed0"], npos=su["npos"], nmom=su["nmom"],
                nisp=su["nisp"]) if n_w and n_w > 0 else None
    nw = n_w if pend is not None else None
    if log_cap is not None:
        stepper = make_pool_stepper(su, cfg, with_seg=True)
        _o, _so, _oo, _pf, logtuple = run_cascade_pool(
            g0, stepper, knuc, su["consumed0"], M=1, max_steps=cfg.max_steps, M_out=24,
            prim_origin=prim_origin, log_cap=log_cap, pending=pend, n_w=nw, q_cap=q_cap,
            per_event_cap=per_event_cap, time_sync=cfg.time_step, step=cfg.step)
        return logtuple
    stepper = make_pool_stepper(su, cfg, with_rec=rec_caps is not None)
    _rc = run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=1, max_steps=cfg.max_steps,
                           M_out=24, prim_origin=prim_origin, rec_caps=rec_caps, pending=pend, n_w=nw,
                           q_cap=q_cap, per_event_cap=per_event_cap, time_sync=cfg.time_step, step=cfg.step)
    if rec_caps is not None:
        out, sofl, oofl, prim_fate, (fsi_rec, _rofl) = _rc
    else:
        out, sofl, oofl, prim_fate = _rc; fsi_rec = None
    sp = out["species"]; chg = out["charge"]; al = out["alive"]; p4o = out["p4"]
    nterms = [dict(species=sp, pid=jnp.where((sp == NUCLEON) & (chg == 1), 2212, 2112),
                   charge=chg, p4=p4o, alive=al, origin=out["origin"], gen=out["gen"],
                   nsc=out["nsc"], track_id=out["track_id"])]
    is_surv_pi = (sp == PION) & al
    if channel == "qe":
        pim = jnp.linalg.norm(p4o[:, :, 1:], axis=2) * is_surv_pi
        jpi = jnp.argmax(pim, axis=1); has_pi = pim[ar, jpi] > 0.0
        created = dict(pid=jnp.where(has_pi, _CH_PID[chg[ar, jpi]], 0), p4=p4o[ar, jpi],
                       w=jnp.ones((n,)), alive=has_pi)
        pterm = dict(species=jnp.zeros((n,), jnp.int32), pid=jnp.zeros((n,), jnp.int32),
                     p4=jnp.zeros((n, 4)), charge=jnp.zeros((n,), jnp.int32), w=jnp.ones((n,)),
                     alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
        return pterm, nterms, sofl + oofl, created, fsi_rec, prim_fate
    is_prim = is_surv_pi & (out["origin"] == _ORIG_PRIM_PI)
    jp = jnp.argmax(is_prim, axis=1); esc_prim = jnp.any(is_prim, axis=1)
    prim_ch = chg[ar, jp]
    _pf0 = prim_fate[:, 0]
    pterm_pid = jnp.where(_pf0 == FATE_ESCAPE, _CH_PID[prim_ch],
                jnp.where(_pf0 == FATE_CONVERT, -1, 0)).astype(jnp.int32)
    pterm = dict(species=jnp.full((n,), PION, jnp.int32), pid=pterm_pid,
                 p4=jnp.where(esc_prim[:, None], p4o[ar, jp], 0.0), charge=prim_ch,
                 w=jnp.ones((n,)), alive=jnp.ones((n,), bool), nsc=jnp.zeros((n,), jnp.int32))
    is_cr = is_surv_pi & (out["origin"] != _ORIG_PRIM_PI)
    cim = jnp.linalg.norm(p4o[:, :, 1:], axis=2) * is_cr
    jc = jnp.argmax(cim, axis=1); has_cr = cim[ar, jc] > 0.0
    created = dict(pid=jnp.where(has_cr, _CH_PID[chg[ar, jc]], 0), p4=p4o[ar, jc],
                   w=jnp.ones((n,)), alive=has_cr)
    return pterm, nterms, sofl + oofl, created, fsi_rec, prim_fate


def cascade_nucleus(p_pi, p_N, pid_pi, pid_Ni, Npid, cfg, key, sabs=1.0, sscat=1.0,
                      channel="res", rec_caps=None, log_cap=None, n_w=None, q_cap=None, per_event_cap=None,
                      su_external=None, return_fate=False):
    """Faithful engine, shared by RES (CC1pi) and QE (CC0pi).
    channel="res": a primary pion segment (+ its top-K knockouts) then a nucleon BFS over {RES recoil,
                   pion knockouts}; pterm = the surviving pion.
    channel="qe":  no primary pion -- gen-0 nucleon = the QE proton (p_N); pterm = "no pion" (pid 0).
    Both share the nucleon BFS + the created-pion (NN->NDelta->Npi) re-entry, so the meson veto (no
    surviving pion for CC0pi / exactly one pi+ for CC1pi) is handled uniformly.
    rec_caps=(Kp,Kn) (pool only): also return the joint per-event kind-1 FSI reweight record as a 5th
    element.  Default None -> 4-tuple.  log_cap=L: run with the in-engine segment logger and return
    (log, counts, overflow).
    Engine defaults (refill + waiting-queue on for every consumer):
      n_w   : None -> refill with working set min(_DEFAULT_NW, n);  0 -> lock-step (bit-exact reference).
      q_cap : None -> _DEFAULT_QCAP (keep overflow particles);  0 -> drop-on-overflow.
      per_event_cap: None -> cfg.max_steps.
    Returns (pterm, nucleon_terminals_per_gen, overflow, created[, fsi_record]) | (log, counts, overflow)."""
    su = setup_nucleus(p_pi, pid_pi, pid_Ni, cfg, key) if su_external is None else su_external
    n = p_pi.shape[0]
    _kpi, knuc, _kpi2 = jax.random.split(su["kp"], 3)
    nw_eff = 0 if n_w == 0 else (min(_DEFAULT_NW, n) if n_w is None else min(int(n_w), n))
    q_eff = _DEFAULT_QCAP if q_cap is None else int(q_cap)
    if log_cap is not None:
        return _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, log_cap=log_cap,
                             n_w=nw_eff, q_cap=q_eff, per_event_cap=per_event_cap)
    res6 = _cascade_pool(channel, p_pi, p_N, Npid, su, cfg, knuc, n, rec_caps=rec_caps,
                         n_w=nw_eff, q_cap=q_eff, per_event_cap=per_event_cap)
    base = res6[:5] if rec_caps is not None else res6[:4]
    return (base + (res6[5],)) if return_fate else base


DiscreteCascadeConfig = CascadeConfig


def pool_cascade_config(**k):
    """The production cascade settings: serial pool engine.

    max_steps is a runaway guard, not a physics bound: with the M=1 serial pool an event's step count
    is the SUM over all its particles, so a small cap trips on ordinary events.  The physics bound is
    path_budget_R * radius.
    """
    return CascadeConfig(step=0.04, max_steps=100000, path_budget_R=20.0, engine="pool", **k)

cascade_nucleus_jit = jax.jit(
    cascade_nucleus,
    static_argnames=("cfg", "channel", "rec_caps", "log_cap", "n_w", "q_cap", "per_event_cap"),
)
