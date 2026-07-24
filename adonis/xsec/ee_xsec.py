"""Absolute ACHILLES inclusive (e,e') cross section + dsigma/domega, by MC over the struck-nucleon
spectral function + isotropic two-body final state, using the SAME ported integrand as the CC QE
driver (adonis/xsec/qe_xsec.py) with three changes for the electromagnetic probe:

  1. BEAM        : monochromatic e- at fixed E (2.222 GeV JLab point), so J_beam = 1 (no flux weight).
  2. PROBE       : me_cross_section(probe="EM", is_proton=...) -> photon leptonic current + the struck
                   nucleon's OWN vector form factors (no axial), spin_avg = 1/4 (2 e- helicities x 2
                   nucleon spins).  See adonis/xsec/dirac.py probe="EM" and electron_scattering.md.
  3. BOTH SPECIES: protons AND neutrons are struck incoherently (CC QE hits neutrons only), each
                   weighted by its target count (Z protons, N neutrons) and its own spectral function.

The observable is inclusive dsigma/domega (omega = E_beam - E'_e) inside the detector angular
acceptance (a HardCut on the outgoing-electron polar angle, exactly the ACHILLES oracle's cut).  The
two-body final state is sampled isotropically in the CM (as ACHILLES's TwoBodyMapper) and the angle
cut is applied as a mask -- so the accepted cross section already carries the acceptance, matching the
oracle's HardCut normalization with no extra factor.

Validation gate: ADoNIS dsigma/domega vs the ACHILLES oracle (output/oracle_ee_C_{qe,res}), ratio +
chi2/ndf, before this sample enters the knob x sample Fisher (electron_scattering.md).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.xsec import constants as C
from adonis.xsec.spectral import SpectralFunction, SpectralImportanceSampler
from adonis.xsec.backend import me_cross_section
from adonis.constants import MASS_PDG_PROTON, MASS_PDG_NEUTRON

_MN = C.mN                       # average nucleon mass (struck-nucleon kinematics; mirrors qe_xsec)
_M_E = 0.51099895               # electron mass [MeV] (ACHILLES Particles.yml); negligible vs 2222 but exact
_TWO_PI = 2 * np.pi
E_BEAM_JLAB = 2222.0            # JLab Murphy:2019wed point [MeV]
THETA_ACC = (14.0, 17.0)        # outgoing-e- polar-angle HardCut [deg] (run_inclusive_ee_C_*.yml)

# per-material target: (Z protons, N neutrons, proton SF, neutron SF)
MATERIALS = {
    "C":  (6, 6,  "data/Spectral_Functions/pke12p_tot.data", "data/Spectral_Functions/pke12n_tot.data"),
    "Ar": (18, 22, "data/Spectral_Functions/pke40p_tot.data", "data/Spectral_Functions/pke40n_tot.data"),
}


def _boost(p4, beta):
    """Boost 4-vector(s) p4 (...,4) by beta (...,3) (CM->lab when beta = P/E of total).  Verbatim qe_xsec."""
    b2 = np.sum(beta ** 2, axis=-1, keepdims=True)
    g = 1.0 / np.sqrt(np.clip(1 - b2, 1e-15, None))
    bp = np.sum(beta * p4[..., 1:], axis=-1, keepdims=True)
    E = (g[..., 0] * (p4[..., 0] + bp[..., 0]))
    fac = (g - 1.0) * np.where(b2 > 1e-15, bp / np.clip(b2, 1e-15, None), 0.0) + g * p4[..., :1]
    vec = p4[..., 1:] + fac * beta
    return np.concatenate([E[..., None], vec], axis=-1)


def _sample_species(n, rng, E_beam, m_species, had_mass, is_proton, sf, n_target):
    """One species: importance-sample the struck nucleon from its SF, isotropic two-body e'+N_out,
    return per-event contribution c (nb) with SUM_i c_i = sigma_species, plus omega and theta_e' [deg].
    Mirrors qe_xsec.sample_importance but for the EM probe + monochromatic e- beam (J_beam = 1)."""
    kz = np.sqrt(E_beam ** 2 - _M_E ** 2)
    k_e = np.tile(np.array([E_beam, 0.0, 0.0, kz]), (n, 1))
    # ---- struck nucleon: |p|,E ~ |p|^2 S importance sampling (the flat-MC weight peak cancels) ----
    samp = SpectralImportanceSampler(sf)
    pvec, E_rm = samp.sample(n, rng)
    p_struck = np.concatenate([(_MN - E_rm)[:, None], pvec], axis=1)      # struck energy: average mN
    # ---- two-body final state e'(m_e) + N_out(m_species), isotropic CM (TwoBodyMapper) ----
    u = rng.random((n, 2))
    P = k_e + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = _M_E ** 2, m_species ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * u[:, 0] - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * u[:, 1]
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    k_e_out = _boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1), beta)
    p_out = _boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1), beta)
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    # ---- removal-energy ceiling (HadronicMapper.cc:50-53), Smin = (m_e + m_species)^2, mono beam ----
    Smin = (_M_E + m_species) ** 2
    mom_s = np.linalg.norm(pvec, axis=1)
    det = E_beam ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * kz + Smin           # pvec.k_e = pz * kz
    emax = _MN + E_beam - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom_s), 400.0)
    valid = (s > Smin) & (lam > 0) & (E_rm > sf.energy[0]) & (E_rm < emax)
    # ---- matrix element (amps2 * FluxFactor * SpinAvg), EM probe, spin_avg = 1/4 ----
    d = me_cross_section(jnp.asarray(k_e), jnp.asarray(k_e_out), jnp.asarray(p_struck),
                         jnp.asarray(p_out), spin_avg=0.25, had_mass=had_mass,
                         probe="EM", is_proton=bool(is_proton))
    me = np.asarray(d["me_xsec"])
    w = np.where(valid, me * n_target * J_2body, 0.0)                     # J_beam = 1 (monochromatic)
    w = np.where(np.isfinite(w), w, 0.0)
    c = w / n                                                            # SUM_i c_i = sigma_species [nb]
    omega = E_beam - k_e_out[:, 0]
    ke_mag = np.linalg.norm(k_e_out[:, 1:], axis=1)
    cos_th = np.clip(k_e_out[:, 3] / np.clip(ke_mag, 1e-9, None), -1, 1)
    theta_deg = np.degrees(np.arccos(cos_th))
    return c, omega, theta_deg, valid


def generate(n, material="C", seed=0, E_beam=E_BEAM_JLAB, chunk=500_000):
    """Inclusive (e,e') MC: n samples per species (p + n).  Returns per-event contribution c [nb]
    (SUM = sigma), omega [MeV], theta_e' [deg], and the struck-species tag.  Chunked to bound JAX mem."""
    Z, N, sf_p_path, sf_n_path = MATERIALS[material]
    sf_p = SpectralFunction(sf_p_path); sf_n = SpectralFunction(sf_n_path)
    out = {"c": [], "omega": [], "theta": [], "is_p": []}
    species = [(True, MASS_PDG_PROTON, MASS_PDG_PROTON, sf_p, Z),
               (False, MASS_PDG_NEUTRON, MASS_PDG_NEUTRON, sf_n, N)]
    for is_p, m_kin, m_flux, sf, n_tgt in species:
        done = 0; sd = seed * 100 + (0 if is_p else 50)
        while done < n:
            m = min(chunk, n - done)
            rng = np.random.default_rng(sd); sd += 1
            # per-species c already normalized by THIS call's m -> rescale to the full-n estimator
            c, omega, theta, valid = _sample_species(m, rng, E_beam, m_kin, m_flux, is_p, sf, n_tgt)
            out["c"].append(c * m / n)                                    # SUM over all chunks = sigma
            out["omega"].append(omega); out["theta"].append(theta)
            out["is_p"].append(np.full(m, is_p))
            done += m
    return {k: np.concatenate(v) for k, v in out.items()}


def dsigma_domega(res, edges, theta_acc=THETA_ACC):
    """Weighted histogram of omega [MeV] inside the theta acceptance -> dsigma/domega [nb/MeV]."""
    lo, hi = theta_acc
    m = (res["theta"] >= lo) & (res["theta"] <= hi)
    h, _ = np.histogram(res["omega"][m], bins=edges, weights=res["c"][m])
    bw = np.diff(edges)
    return h / bw


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    mat = sys.argv[2] if len(sys.argv) > 2 else "C"
    r = generate(n, material=mat)
    lo, hi = THETA_ACC
    acc = (r["theta"] >= lo) & (r["theta"] <= hi)
    sig_tot = r["c"].sum()
    sig_acc = r["c"][acc].sum()
    print(f"(e,e') {mat}  n={n}/species  sigma_total={sig_tot:.4e} nb  "
          f"sigma_in_[{lo},{hi}]deg={sig_acc:.4e} nb  (accepted {acc.sum()} ev, "
          f"{100*acc.mean():.2f}% of samples)")
    edges = np.linspace(0, 600, 31)
    dsdo = dsigma_domega(r, edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    pk = ctr[np.argmax(dsdo)]
    print(f"  dsigma/domega peak at omega ~ {pk:.0f} MeV,  max = {dsdo.max():.3e} nb/MeV")
