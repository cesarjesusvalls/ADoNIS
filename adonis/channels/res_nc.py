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
