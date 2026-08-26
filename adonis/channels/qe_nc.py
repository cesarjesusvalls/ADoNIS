"""Neutral-current QUASI-ELASTIC scattering: nu N -> nu N, on a free nucleon.

The NC twin of qe.py, kept separate rather than a `probe=` flag on it because qe.sample_importance
hardcodes the outgoing muon mass and the `n -> mu- p` channel.

NC elastic differs from CC QE in more than the coupling:

  * BOTH nucleons participate: CC QE is `nu n -> mu- p` only; NC has `nu p -> nu p` AND `nu n -> nu n`.
  * The final nucleon is the SAME species as the initial one, so there is no mass step.
  * The outgoing lepton is massless.

Cross-checked against ACHILLES (`configs/achilles/run_freenucleon_nc_qe_{H,N}.yml`, free nucleon, no
cascade), proc IDs 250 (`nu n -> nu n`) and 251 (`nu p -> nu p`).  The n/p cross-section ratio is the
sharpest available isospin check and needs no normalisation convention to be right: the proton's NC
vector coupling carries (1/2 - 2 sin^2 th_W) while the neutron's carries -1/2, so the proton's NC
elastic is axial-dominated and the neutron's is not.

Two-body phase space, nucleon at rest, sampled isotropically in the CM:

    sigma = <me_xsec>_{Omega* uniform} * Phi_2 ,   Phi_2 = |p*| / (4 pi sqrt(s))

with me_xsec = amps2 * flux_factor * spin_avg from matrix_element.me_cross_section (i.e. everything
except the phase-space Jacobian).
"""
from __future__ import annotations

import numpy as np

from adonis.channels import constants as C
from adonis.channels.twobody import isotropic_two_body_cm
from adonis.channels.currents.matrix_element import (me_cross_section, MASS_PDG_PROTON,
                                                     MASS_PDG_NEUTRON)
from adonis.nuclear.targets import spectral_inputs
from adonis.flux.spectrum import SpectrumFlux
from adonis.nuclear.spectral import SpectralFunction, SpectralImportanceSampler

SPIN_AVG_NC = 0.5
_MN = C.mN
_TWO_PI = 2 * np.pi


def _two_body_cm(k_nu, m_N, u):
    """nu(k) + N(at rest) -> nu(k') + N(p'), isotropic in the CM.  Returns (k_lep, p_out, Phi2).

    A rest-frame special case (massless outgoing nu, elastic nucleon) of the shared two-body decay: the
    total 4-momentum is k_nu + N-at-rest, m1=0, m2=m_N.  The general boost reduces exactly to the former
    hand-written +z boost because P is along +z here (validated against the G6(2) free-nucleon sigma)."""
    n = len(k_nu)
    P = k_nu + np.column_stack([np.full(n, m_N), np.zeros((n, 3))])
    k_lep, p_out, pcm, sqs, _s, _lam = isotropic_two_body_cm(P, 0.0, m_N, u[:, 0], u[:, 1])
    phi2 = pcm / (4.0 * np.pi * sqs)
    return k_lep, p_out, phi2


def sigma_free_nucleon_nc_qe(Enu_MeV, is_proton, n=200_000, seed=0, use_achilles_nc_coupling=False, chunk=100_000):
    """Free-nucleon NC elastic sigma(E_nu) [nb] + standard error, nucleon at rest.

    `use_achilles_nc_coupling`: False = correct physics (the SM coupling); True = ACHILLES's coupl1
    convention verbatim, for a like-for-like comparison against ACHILLES output.
    """
    m_N = MASS_PDG_PROTON if is_proton else MASS_PDG_NEUTRON
    rng = np.random.default_rng(seed)
    E = float(Enu_MeV)
    k_nu = np.column_stack([np.full(n, E), np.zeros(n), np.zeros(n), np.full(n, E)])
    p_in = np.column_stack([np.full(n, m_N), np.zeros(n), np.zeros(n), np.zeros(n)])
    u = rng.random((n, 2))
    k_lep, p_out, phi2 = _two_body_cm(k_nu, m_N, u)
    isp = np.full(n, bool(is_proton))
    w = np.zeros(n)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        d = me_cross_section(k_nu[sl], k_lep[sl], p_in[sl], p_out[sl], spin_avg=SPIN_AVG_NC,
                             had_mass=m_N, probe="NC", is_proton=isp[sl], use_achilles_nc_coupling=use_achilles_nc_coupling)
        w[sl] = np.asarray(d["me_xsec"]) * phi2[sl]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return float(w.mean()), float(w.std() / np.sqrt(n))


def _sample_species_nc(n, rng, flux, minE, m_species, is_proton, sf, n_target, use_achilles_nc_coupling):
    """One struck-nucleon species: flux beam + SF-importance-sampled struck nucleon + isotropic
    nu' + N_out (NC weak vertex).  Returns per-event RAW weight w (mean over n_target-weighted draws ->
    sigma_species) and lab momenta.  Massless outgoing neutrino (m1=0), elastic nucleon (m2 = m_species),
    so the two-body threshold is Smin = m_species^2.  Mirrors ee._sample_species (both species, species
    mass) + qe.sample_importance (flux + SF + removal-energy ceiling), with the NC current from the
    free-nucleon function above."""
    u = rng.random((n, 3))
    E_GeV, J_beam = flux.sample_beam(u[:, 0], minE)
    Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    pvec, E_rm = SpectralImportanceSampler(sf).sample(n, rng)
    p_struck = np.concatenate([(m_species - E_rm)[:, None], pvec], axis=1)
    P = k_nu + p_struck
    k_lep, p_out, pcm, sqrts, s, lam = isotropic_two_body_cm(P, 0.0, m_species, u[:, 1], u[:, 2])
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    Smin = m_species ** 2
    mom_s = np.linalg.norm(pvec, axis=1)
    det = Enu ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = _MN + Enu - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom_s), 400.0)
    valid = (s > Smin) & (lam > 0) & (E_rm > sf.energy[0]) & (E_rm < emax)
    d = me_cross_section(k_nu, k_lep, p_struck, p_out, spin_avg=SPIN_AVG_NC, had_mass=m_species,
                         probe="NC", is_proton=bool(is_proton), use_achilles_nc_coupling=use_achilles_nc_coupling)
    me = np.asarray(d["me_xsec"])
    w = np.where(valid, me * n_target * J_2body * J_beam, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    pid = 2212 if is_proton else 2112
    return dict(w=w, k_nu=k_nu, k_lep=k_lep, p_struck=p_struck, p_out=p_out,
                pid=np.full(n, pid, np.int32))


def generate(n, material="C", seed=0, chunk=500_000, return_events=False, use_achilles_nc_coupling=False, theta_acc=None):
    """Nuclear NC-QE generator: nu N -> nu N' on `material`, n draws SPLIT across BOTH struck species
    (protons AND neutrons -- NC is elastic on either, unlike CC nu n -> mu p).  Flux-averaged over the
    beam (ADONIS_FLUX_FILE, set per-bank by generate_bank).  Per-event weight w already carries the MC
    norm so sum(w) = flux-averaged nuclear NC-QE sigma [nb].  return_events=True returns the bank fields
    the generate_bank NC branch consumes (k_lep = outgoing neutrino, p_N = outgoing nucleon, no pion).

    NO theta_acc: a polar cut on the INVISIBLE outgoing neutrino is meaningless (refused, like res_nc)."""
    if theta_acc is not None and tuple(float(x) for x in theta_acc) != (0.0, 180.0):
        raise ValueError("NC QE: refusing a polar-angle cut on the invisible outgoing neutrino")
    Z, N, sf_p_path, sf_n_path = spectral_inputs(material)
    sf_p = SpectralFunction(sf_p_path); sf_n = SpectralFunction(sf_n_path)
    flux = SpectrumFlux()
    minE = flux.seed_min_GeV(m_lep=0.0)
    species = [(True, MASS_PDG_PROTON, sf_p, Z), (False, MASS_PDG_NEUTRON, sf_n, N)]
    cols = ("w", "k_nu", "k_lep", "p_struck", "p_out", "pid")
    acc = {k: [] for k in cols}
    for si, (is_p, m_sp, sf, n_tgt) in enumerate(species):
        n_s = n // 2 + (1 if si < n % 2 else 0)
        done = 0; sd = seed * 100 + (0 if is_p else 50)
        while done < n_s:
            m = min(chunk, n_s - done)
            rng = np.random.default_rng(sd); sd += 1
            r = _sample_species_nc(m, rng, flux, minE, m_sp, is_p, sf, n_tgt, use_achilles_nc_coupling)
            r["w"] = r["w"] / n_s
            for k in cols:
                acc[k].append(r[k])
            done += m
    B = {k: np.concatenate(v) for k, v in acc.items()}
    sigma = float(B["w"].sum())
    if not return_events:
        return {"sigma": sigma}
    nev = len(B["w"])
    ev = dict(w=B["w"], k_nu=B["k_nu"].astype(np.float32), p_struck=B["p_struck"].astype(np.float32),
              k_lep=B["k_lep"].astype(np.float32), p_N=B["p_out"].astype(np.float32),
              p_pi=np.zeros((nev, 4), np.float32), ppid=np.zeros(nev, np.int32),
              ipid=B["pid"].astype(np.int32),
              Npid=B["pid"].astype(np.int32))
    return {"events": ev, "sigma": sigma}
