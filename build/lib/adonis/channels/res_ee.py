"""Inclusive (e,e') RES (single-pion) cross section + dsigma/domega, mirroring adonis/channels/res.py but
for the ELECTROMAGNETIC probe.  Three changes vs the CC RES driver:

  1. BEAM   : monochromatic e- at fixed E (J_beam = 1).
  2. PROBE  : exclusive_amps2_batch(probe="EM") -> DCC mode=10 (axial off, EM N->Delta isospin: proton
              -> vec, neutron I=1/2 -> -isv isoscalar) + photon leptonic current + i/q^2 + _NORM_EM.
              spin_avg = 1/4 (2 e- helicities x 2 nucleon spins).
  3. CHANNELS: the 4 EM 1-pi channels {p->p pi0, p->n pi+, n->n pi0, n->p pi-}, each struck nucleon
              drawn from its OWN spectral function (imp_p / imp_n) so the target count Z/N is explicit.

Observable: inclusive dsigma/domega (omega = E_beam - E'_e) inside the outgoing-electron theta HardCut,
summed over channels.
"""
from __future__ import annotations

import numpy as np

from adonis.channels import constants as C
from adonis.nuclear.targets import spectral_inputs
from adonis.nuclear.spectral import SpectralFunction, SpectralImportanceSampler
from adonis.channels.currents.matrix_element import flux_factor, MASS_PDG_PROTON, MASS_PDG_NEUTRON
from adonis.channels.dcc.current import exclusive_amps2_batch
from adonis.channels.res import _boost_to_lab, _sqlam, M_PIP, M_PI0

_MN = C.mN
from adonis.flux.electron import M_E as _M_E, E_BEAM_JLAB, electron_k
_TWO_PI = 2 * np.pi
THETA_ACC = (14.0, 17.0)
SPIN_AVG_EM = 0.25

EM_CHANNELS = [
    (2212, +1, 111, MASS_PDG_PROTON,  True),
    (2212, +1, 211, MASS_PDG_NEUTRON, True),
    (2112, -1, 111, MASS_PDG_NEUTRON, False),
    (2112, -1, -211, MASS_PDG_PROTON, False),
]
_M_PI = {111: M_PI0, 211: M_PIP, -211: M_PIP}



def _sample_3body_ee(k_e, p_struck, m_pi, m_Nf, m_lep, u):
    """3-body final state e' + N + pi via two isotropic 2-body splits.  Verbatim res._sample_3body
    with the outgoing-lepton mass generalized to m_lep (electron, not muon)."""
    P = k_e + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - m_pi) ** 2; s23min = np.maximum((m_lep + m_Nf) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 0]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    EleN = (s + s23 - m_pi ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, s23, m_pi ** 2) / 2
    ctA = 2 * u[:, 1] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 2]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    leN_cm = np.concatenate([EleN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(m_pi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_leN = _boost_to_lab(leN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, m_pi ** 2), 1e-12, None)
    Ele = (s23 + m_lep ** 2 - m_Nf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, m_lep ** 2, m_Nf ** 2) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    le_cm = np.concatenate([Ele[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(m_Nf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_lep = _boost_to_lab(le_cm, p_leN); p_N = _boost_to_lab(N_cm, p_leN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, m_lep ** 2, m_Nf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J_3body = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid3 = ((s23max > s23min) & (_sqlam(s, s23, m_pi ** 2) > 0)
              & (_sqlam(s23, m_lep ** 2, m_Nf ** 2) > 0))
    return dict(k_lep=k_lep, p_N=p_N, p_pi=p_pi, J_3body=J_3body, s=s, valid3=valid3)


def _sample_channel_ee(n, rng, E_beam, m_pi, m_Nf, m_struck, imp):
    kz = np.sqrt(E_beam ** 2 - _M_E ** 2)
    k_e = electron_k(E_beam, n)
    u = rng.random((n, 10))
    pvec, energy = imp.sample(n, rng)
    mom = np.linalg.norm(pvec, axis=1)
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    tb = _sample_3body_ee(k_e, p_struck, m_pi, m_Nf, _M_E, u[:, 5:10])
    Smin = (_M_E + m_Nf + m_pi) ** 2
    det = E_beam ** 2 + mom ** 2 + 2 * pvec[:, 2] * kz + Smin
    emax = _MN + E_beam - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom), 400.0)
    valid = (tb["s"] > Smin) & tb["valid3"] & (energy < emax)
    return dict(k_e=k_e, p_struck=p_struck, k_lep=tb["k_lep"], p_N=tb["p_N"], p_pi=tb["p_pi"],
                J=tb["J_3body"], mom=mom, energy=energy, valid=valid)


def generate(n, material="C", seed=0, E_beam=E_BEAM_JLAB, chunk=250_000, records=False, theta_acc=None):
    """Inclusive (e,e') RES MC: n TOTAL samples, split across the 4 EM channels.  Returns per-event contribution c [nb]
    (SUM = sigma), omega [MeV], theta_e' [deg].
    records=True returns the hadronic final state for the kept events -- p_N, p_pi [4-mom], ppid (pion),
    Npid (final nucleon), ipid (struck nucleon) -- everything cascade_nucleus needs to run the recorded FSI
    cascade (mirrors res.generate(return_events=True) for the neutrino bank).
    theta_acc=(lo,hi) [deg] applies the SAME outgoing-e- angular acceptance the QE channel (ee.generate)
    applies -- both EM channels then honor the config's acceptance identically (default None = all angles,
    kept for the raw dsigma/domega estimator; the bank generator always passes cfg.theta_acc)."""
    Z, N, sf_p_path, sf_n_path = spectral_inputs(material)
    imp_p = SpectralImportanceSampler(SpectralFunction(sf_p_path))
    imp_n = SpectralImportanceSampler(SpectralFunction(sf_n_path))
    keys = ["c", "omega", "theta"] + (["k_e", "k_lep", "p_struck", "p_N", "p_pi",
                                        "ppid", "Npid", "ipid"] if records else [])
    out = {k: [] for k in keys}
    nch = len(EM_CHANNELS)
    for ci, (spid, itiz, ppid, m_Nf, is_p) in enumerate(EM_CHANNELS):
        imp = imp_p if is_p else imp_n
        n_tgt = Z if is_p else N
        m_struck = MASS_PDG_PROTON if is_p else MASS_PDG_NEUTRON
        Npid = 2212 if m_Nf == MASS_PDG_PROTON else 2112
        m_pi = _M_PI[ppid]
        n_ch = n // nch + (1 if ci < n % nch else 0)
        done = 0; sd = seed * 1000 + ci * 100
        while done < n_ch:
            m = min(chunk, n_ch - done)
            rng = np.random.default_rng(sd); sd += 1
            s = _sample_channel_ee(m, rng, E_beam, m_pi, m_Nf, m_struck, imp)
            a2 = np.zeros(m); v = s["valid"]
            if v.any():
                a2[v] = exclusive_amps2_batch(s["k_e"][v], s["k_lep"][v], s["p_struck"][v],
                                              s["p_N"][v], s["p_pi"][v], itiz, ppid, probe="EM", tcrz=0.0)
            fl = np.asarray(flux_factor(s["k_e"], s["p_struck"], had_mass=m_struck))
            w = np.where(v, a2 * fl * n_tgt * SPIN_AVG_EM * s["J"], 0.0)
            w = np.where(np.isfinite(w), w, 0.0)
            k_lep = s["k_lep"]
            omega = E_beam - k_lep[:, 0]
            kmag = np.linalg.norm(k_lep[:, 1:], axis=1)
            theta = np.degrees(np.arccos(np.clip(k_lep[:, 3] / np.clip(kmag, 1e-9, None), -1, 1)))
            if records:
                sel = v if theta_acc is None else (v & (theta >= theta_acc[0]) & (theta <= theta_acc[1]))
                out["c"].append((w / n_ch)[sel]); out["omega"].append(omega[sel]); out["theta"].append(theta[sel])
                out["k_e"].append(s["k_e"][sel]); out["k_lep"].append(s["k_lep"][sel])
                out["p_struck"].append(s["p_struck"][sel])
                out["p_N"].append(s["p_N"][sel]); out["p_pi"].append(s["p_pi"][sel])
                nk = int(sel.sum())
                out["ppid"].append(np.full(nk, ppid, np.int32))
                out["Npid"].append(np.full(nk, Npid, np.int32))
                out["ipid"].append(np.full(nk, spid, np.int32))
            else:
                out["c"].append(w / n_ch)
                out["omega"].append(omega); out["theta"].append(theta)
            done += m
    return {k: np.concatenate(v) for k, v in out.items()}


def dsigma_domega(res, edges, theta_acc=THETA_ACC):
    lo, hi = theta_acc
    mth = (res["theta"] >= lo) & (res["theta"] <= hi)
    h, _ = np.histogram(res["omega"][mth], bins=edges, weights=res["c"][mth])
    return h / np.diff(edges)


from adonis.core.sample import Sampler


class RESEEChannel(Sampler):
    """Inclusive (e,e') RES single-pion (EM probe, monochromatic e- beam) as a detached kind-1 SAMPLER.
    `propose(seed, n)` = `generate(records=True, theta_acc=...)` -- n events, angular acceptance applied
    (the SAME contract as EEChannel, so both (e,e') channels are homogeneous)."""

    def __init__(self, material="C", e_beam=E_BEAM_JLAB, theta_acc=THETA_ACC):
        self.material = material; self.e_beam = e_beam; self.theta_acc = theta_acc

    def propose(self, key, n):
        return generate(n, material=self.material, seed=int(key), E_beam=self.e_beam,
                        records=True, theta_acc=self.theta_acc)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    mat = sys.argv[2] if len(sys.argv) > 2 else "C"
    r = generate(n, material=mat)
    lo, hi = THETA_ACC
    acc = (r["theta"] >= lo) & (r["theta"] <= hi)
    print(f"(e,e') RES {mat}  n={n}/channel  sigma_total={r['c'].sum():.4e} nb  "
          f"sigma_in_[{lo},{hi}]deg={r['c'][acc].sum():.4e} nb  ({acc.sum()} accepted)")
    edges = np.linspace(0, 800, 41)
    dsdo = dsigma_domega(r, edges); ctr = 0.5 * (edges[:-1] + edges[1:])
    print(f"  dsigma/domega peak at omega ~ {ctr[np.argmax(dsdo)]:.0f} MeV, max = {dsdo.max():.3e} nb/MeV")
