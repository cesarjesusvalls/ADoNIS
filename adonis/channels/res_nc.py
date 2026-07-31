"""Neutral-current single-pion RES: nu N -> nu N pi.

Fourth sibling of res.py (CC) / res_ee.py (EM), and deliberately built on res_ee rather than res:

  * res.py hardcodes M_MU in SEVENTEEN places inside its 3-body sampler, so "reuse it with m_lep=0"
    is not a small edit -- it is a rewrite with seventeen chances to miss one.
  * res_ee.py ALREADY parametrises the outgoing-lepton mass (`_sample_3body_ee(..., m_lep, u)`),
    already carries a 4-channel table, and already passes tcrz=0.  Reusing that sampler with
    m_lep = 0 is a one-argument change to code that is validated for a NON-muon lepton.

What differs from EM (verified against amp_dcc_sl_module.f, read directly):

  probe      : "NC" -> DCC mode = -1, NC leptonic current, _NORM_NC   (vs mode = 10, photon, _NORM_EM)
  spin_avg   : 1/2  -- ONE neutrino helicity x 2 nucleon spins        (vs 1/4 for the electron)
  m_lep      : 0                                                      (vs m_e)
  axial      : PRESENT (`if(mode.lt.10)` gates the axial block :802)  (vs absent for EM)
  pion pole  : ABSENT (`if(mode.gt.0)`, CC modes 1-4 only, :844)      (same as EM)
  isospin    : no sqrt(2) factor (`mode 1..4` only, :275)             (same as EM)
  vector     : VFAC*vec + (I==1/2)*VVFAC(itiz)*isv, sw2 = 0.2312      (vs raw vec / -isv)

The four channels are the same four as EM -- NC cannot change the nucleon charge, so the pion charge
is fixed by the final nucleon.  ACHILLES confirms them by running:
`configs/achilles/run_freenucleon_nc_res_{H,N}.yml` produce exactly
    on a proton: p -> p pi0 (62.7%),  p -> n pi+ (37.3%)
    on a neutron: n -> n pi0 (62.4%),  n -> p pi- (37.6%)
under proc IDs 451/452.  The 0.3% p/n mirror symmetry is the VVFAC sign flip.

`sigma_free_nucleon_nc` is the ADoNIS side of gate G5(2): absolute sigma(E_nu) in nb, to be compared
against those cards WITHOUT any bridge constant.  A constant offset means a coupling or _NORM error;
an energy slope means a propagator error -- report mean and spread SEPARATELY, never a single number.
"""
from __future__ import annotations

import numpy as np

from adonis.channels import constants as C
from adonis.channels.currents.matrix_element import flux_factor, MASS_PDG_PROTON, MASS_PDG_NEUTRON
from adonis.channels.dcc.current import exclusive_amps2_batch
from adonis.channels.res import _pi_kin_mass, M_PIP, M_PI0
from adonis.channels.res_ee import _sample_3body_ee

_MN = C.mN
M_LEP_NC = 0.0                  # the outgoing lepton is a neutrino
SPIN_AVG_NC = 0.5               # 1 neutrino helicity x 2 nucleon spins (CC's value, not EM's 1/4)

# (struck pid, itiz, pion pid, m_Nf [MeV], is_proton_struck) -- same four as EM_CHANNELS.
NC_RES_CHANNELS = [
    (2212, +1, 111, MASS_PDG_PROTON,  True),     # p -> p pi0   (ACHILLES proc 452, 62.7% on 1H)
    (2212, +1, 211, MASS_PDG_NEUTRON, True),     # p -> n pi+   (ACHILLES proc 451, 37.3% on 1H)
    (2112, -1, 111, MASS_PDG_NEUTRON, False),    # n -> n pi0   (ACHILLES proc 451, 62.4% on 1N)
    (2112, -1, -211, MASS_PDG_PROTON, False),    # n -> p pi-   (ACHILLES proc 452, 37.6% on 1N)
]
_M_PI = {111: M_PI0, 211: M_PIP, -211: M_PIP}


def free_nucleon_weights_nc(k_nu, itiz, m_Nf, pi_pid, m_pi_phys, had_mass, u, chunk=50_000):
    """Per-event FREE-nucleon NC single-pion weight w = amps2 * flux * SPIN_AVG_NC * J_3body, for a
    nucleon AT REST and NO beam Jacobian -- the NC twin of res.free_nucleon_weights.

    Kept as its own function rather than a `probe=` argument on the CC one: that primitive routes
    through res._sample_3body_dispatch, which is the M_MU-hardcoded sampler.  Two short functions
    that each say what they mean beat one that silently depends on a module-level lepton mass.

    tcrz = 0 for the NC current (amp_dcc_sl_module.f:284-285), exactly as for the photon.  With the
    CC default of 1 the isospin Clebsch-Gordan <1,tcrz;1/2,tiz|tpi,tpiz> kills the pi0 channels --
    which are 62% of NC RES, so the failure would be loud but the cause obscure.
    """
    n = len(k_nu)
    m_pi = _pi_kin_mass(m_pi_phys)
    p_struck = np.tile([had_mass, 0.0, 0.0, 0.0], (n, 1)).astype(float)      # nucleon at rest
    tb = _sample_3body_ee(k_nu, p_struck, m_pi, m_Nf, M_LEP_NC, u)
    k_lep, p_N, p_pi, J3, valid = tb["k_lep"], tb["p_N"], tb["p_pi"], tb["J_3body"], tb["valid3"]
    a2 = np.zeros(n)
    idx = np.where(valid & (J3 > 0))[0]
    for i in range(0, len(idx), chunk):
        sl = idx[i:i + chunk]
        a2[sl] = np.asarray(exclusive_amps2_batch(k_nu[sl], k_lep[sl], p_struck[sl], p_N[sl],
                                                  p_pi[sl], int(itiz), int(pi_pid),
                                                  probe="NC", tcrz=0.0))
    fl = np.asarray(flux_factor(k_nu, p_struck, had_mass=had_mass))
    w = np.where(valid, a2 * fl * SPIN_AVG_NC * J3, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return w, dict(k_lep=k_lep, p_N=p_N, p_pi=p_pi, valid=valid)


def sigma_free_nucleon_nc(Enu_MeV, channel, n=80_000, seed=0):
    """Monochromatic free-nucleon NC single-pion sigma(E_nu) [nb] + standard error (J_beam = 1, so
    sigma = <w>).  `channel` is a row of NC_RES_CHANNELS.

    This is the ADoNIS half of G5(2).  Compare against configs/achilles/run_freenucleon_nc_res_*.yml
    in ABSOLUTE nb: no bridge constant, and report mean(ACH/ADO) and spread(ACH/ADO) apart -- a flat
    offset is a coupling/_NORM error, a slope in E is a propagator error, and a single averaged
    number cannot tell them apart.
    """
    _pdg_in, itiz, pi_pid, m_Nf, _is_p = channel
    had_mass = MASS_PDG_PROTON if _is_p else MASS_PDG_NEUTRON
    rng = np.random.default_rng(seed)
    u = rng.random((n, 6))                       # col 0 unused: keeps the res.free_proton u-stream
    E = float(Enu_MeV)
    k_nu = np.stack([np.full(n, E), np.zeros(n), np.zeros(n), np.full(n, E)], axis=1)
    w, _ = free_nucleon_weights_nc(k_nu, itiz, m_Nf, pi_pid, _M_PI[pi_pid], had_mass, u[:, 1:6])
    return float(w.mean()), float(w.std() / np.sqrt(n))


def sigma_free_nucleon_nc_total(Enu_MeV, is_proton, n=80_000, seed=0):
    """Total free-nucleon NC RES sigma(E_nu) [nb] on a proton or a neutron: the sum over that
    nucleon's two channels, which is what an ACHILLES `1H` / `1N` card reports."""
    tot = 0.0
    var = 0.0
    for ci, ch in enumerate(c for c in NC_RES_CHANNELS if c[4] == is_proton):
        s, e = sigma_free_nucleon_nc(Enu_MeV, ch, n=n, seed=seed * 100 + ci)
        tot += s
        var += e ** 2
    return tot, float(np.sqrt(var))


# =========================================================================== nucleus-level generator
MATERIALS = {
    "C":  (6, 6,   "data/Spectral_Functions/pke12p_tot.data", "data/Spectral_Functions/pke12n_tot.data"),
    "Ar": (18, 22, "data/Spectral_Functions/pke40p_tot.data", "data/Spectral_Functions/pke40n_tot.data"),
}


def _sample_channel_nc(n, rng, flux, minE, maxE, m_pi, m_Nf, had_mass, imp):
    """One NC RES channel on a BOUND nucleon: spectrum beam + |p|^2 S importance-sampled struck
    nucleon + the shared 3-body core.  Mirrors res_ee._sample_channel_ee with the monochromatic
    electron beam replaced by a flux draw, and m_lep = 0."""
    u = rng.random((n, 10))
    E_GeV = u[:, 4] * (maxE - minE) + minE
    Enu = E_GeV * 1000.0
    J_beam = ((maxE - minE) * flux.f(E_GeV)) / flux.flux_integral
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    pvec, energy = imp.sample(n, rng)                     # |p|^2 S importance (initwgt -> N constant)
    mom = np.linalg.norm(pvec, axis=1)
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    tb = _sample_3body_ee(k_nu, p_struck, m_pi, m_Nf, M_LEP_NC, u[:, 5:10])
    Smin = (M_LEP_NC + m_Nf + m_pi) ** 2                  # no lepton-mass term: NC has no threshold
    det = Enu ** 2 + mom ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = _MN + Enu - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom), 400.0)
    valid = (tb["s"] > Smin) & tb["valid3"] & (energy < emax)
    return dict(k_nu=k_nu, p_struck=p_struck, k_lep=tb["k_lep"], p_N=tb["p_N"], p_pi=tb["p_pi"],
                J=tb["J_3body"] * J_beam, valid=valid)


def generate(n=20000, material="C", seed=0, return_events=False, chunk=250_000,
             n_neutron=None, n_proton=None, theta_acc=None):
    """Flux-averaged NC single-pion RES on a NUCLEUS.  n is the TOTAL draw count, split across the
    four channels (the same stratified convention as res.generate_importance and res_ee.generate).

    Per-species spectral functions with EXPLICIT Z / N target counting -- the isoscalar term makes NC
    genuinely p/n-asymmetric, so a shared count would bias the pi0 fraction, which IS the signal.

    theta_acc is accepted and must be full acceptance: a polar cut on an invisible outgoing neutrino
    is meaningless, and silently applying one would bias the sample with nothing to notice it.
    """
    from adonis.flux.spectrum import SpectrumFlux
    from adonis.nuclear.spectral import SpectralFunction, SpectralImportanceSampler
    if theta_acc is not None and not (theta_acc[0] <= 0.0 and theta_acc[1] >= 180.0):
        raise ValueError(f"probe=NC cannot honour theta_acc={theta_acc}: the outgoing lepton is a "
                         "neutrino, so a polar acceptance on it is meaningless")
    Z, N, sf_p_path, sf_n_path = MATERIALS[material]
    n_proton = Z if n_proton is None else n_proton
    n_neutron = N if n_neutron is None else n_neutron
    imp_p = SpectralImportanceSampler(SpectralFunction(sf_p_path))
    imp_n = SpectralImportanceSampler(SpectralFunction(sf_n_path))
    flux = SpectrumFlux()
    minE = flux.seed_min_GeV(m_lep=M_LEP_NC); maxE = flux.max_energy   # NC has no lepton threshold
    ev = {k: [] for k in ("k_nu", "k_lep", "p_struck", "p_N", "p_pi", "w", "ppid", "Npid", "ipid")}
    sig = 0.0
    nch = len(NC_RES_CHANNELS)
    for ci, (spid, itiz, ppid, m_Nf, is_p) in enumerate(NC_RES_CHANNELS):
        imp = imp_p if is_p else imp_n
        n_tgt = n_proton if is_p else n_neutron
        had_mass = MASS_PDG_PROTON if is_p else MASS_PDG_NEUTRON
        Npid = 2212 if m_Nf == MASS_PDG_PROTON else 2112
        m_pi = _pi_kin_mass(_M_PI[ppid])
        n_ch = n // nch + (1 if ci < n % nch else 0)
        done = 0; sd = seed * 1000 + ci * 100
        while done < n_ch:
            m = min(chunk, n_ch - done)
            rng = np.random.default_rng(sd); sd += 1
            s = _sample_channel_nc(m, rng, flux, minE, maxE, m_pi, m_Nf, had_mass, imp)
            a2 = np.zeros(m); v = s["valid"]
            if v.any():
                a2[v] = np.asarray(exclusive_amps2_batch(
                    s["k_nu"][v], s["k_lep"][v], s["p_struck"][v], s["p_N"][v], s["p_pi"][v],
                    itiz, ppid, probe="NC", tcrz=0.0))
            fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=had_mass))
            w = np.where(v, a2 * fl * n_tgt * SPIN_AVG_NC * s["J"], 0.0)
            w = np.where(np.isfinite(w), w, 0.0) / n_ch
            sig += w.sum()
            if return_events:
                keep = w > 0
                nk = int(keep.sum())
                for k in ("k_nu", "k_lep", "p_struck", "p_N", "p_pi"):
                    ev[k].append(s[k][keep])
                ev["w"].append(w[keep])
                ev["ppid"].append(np.full(nk, ppid, np.int32))
                ev["Npid"].append(np.full(nk, Npid, np.int32))
                ev["ipid"].append(np.full(nk, spid, np.int32))
            done += m
    if not return_events:
        return dict(sigma=sig)
    return dict(sigma=sig, events={k: np.concatenate(v) if v else np.zeros((0, 4)) for k, v in ev.items()})
