"""Absolute ACHILLES inclusive (e,e') cross section + dsigma/domega, by MC over the struck-nucleon
spectral function + isotropic two-body final state, using the SAME ported integrand as the CC QE
driver (adonis/channels/qe.py) with three changes for the electromagnetic probe:

  1. BEAM        : monochromatic e- beam at fixed E, so J_beam = 1 (no flux weight).
  2. PROBE       : me_cross_section(probe="EM", is_proton=...) -> photon leptonic current + the struck
                   nucleon's OWN vector form factors (no axial), spin_avg = 1/4 (2 e- helicities x 2
                   nucleon spins).  See adonis/channels/dirac.py probe="EM".
  3. BOTH SPECIES: protons AND neutrons are struck incoherently (CC QE hits neutrons only), each
                   weighted by its target count (Z protons, N neutrons) and its own spectral function.

The observable is inclusive dsigma/domega (omega = E_beam - E'_e) inside the detector angular
acceptance (a HardCut on the outgoing-electron polar angle, matching the ACHILLES oracle's cut).  The
two-body final state is sampled isotropically in the CM, and the angle cut is applied as a mask, so
the accepted cross section already carries the acceptance with no extra normalization factor.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.channels import constants as C
from adonis.nuclear.targets import spectral_inputs
from adonis.nuclear.spectral import SpectralFunction, SpectralImportanceSampler
from adonis.channels.currents.matrix_element import me_cross_section
from adonis.constants import MASS_PDG_PROTON, MASS_PDG_NEUTRON

_MN = C.mN
from adonis.flux.electron import M_E as _M_E, E_BEAM_JLAB, electron_k
_TWO_PI = 2 * np.pi
THETA_ACC = (14.0, 17.0)



from adonis.channels.twobody import isotropic_two_body_cm


STRUCK_MASS_MODE = "species"


def _sample_species(n, rng, E_beam, m_species, had_mass, is_proton, sf, n_target):
    """One species: importance-sample the struck nucleon from its SF, isotropic two-body e'+N_out,
    return per-event contribution c (nb) with SUM_i c_i = sigma_species, plus omega and theta_e' [deg].
    Mirrors qe.sample_importance but for the EM probe + monochromatic e- beam (J_beam = 1)."""
    kz = np.sqrt(E_beam ** 2 - _M_E ** 2)
    k_e = electron_k(E_beam, n)
    samp = SpectralImportanceSampler(sf)
    pvec, E_rm = samp.sample(n, rng)
    m_struck = m_species if STRUCK_MASS_MODE == "species" else _MN
    p_struck = np.concatenate([(m_struck - E_rm)[:, None], pvec], axis=1)
    u = rng.random((n, 2))
    P = k_e + p_struck
    k_lep, p_out, pcm, sqrts, s, lam = isotropic_two_body_cm(P, _M_E, m_species, u[:, 0], u[:, 1])
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    Smin = (_M_E + m_species) ** 2
    mom_s = np.linalg.norm(pvec, axis=1)
    det = E_beam ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * kz + Smin
    emax = _MN + E_beam - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom_s), 400.0)
    valid = (s > Smin) & (lam > 0) & (E_rm > sf.energy[0]) & (E_rm < emax)
    d = me_cross_section(jnp.asarray(k_e), jnp.asarray(k_lep), jnp.asarray(p_struck),
                         jnp.asarray(p_out), spin_avg=0.25, had_mass=had_mass,
                         probe="EM", is_proton=bool(is_proton))
    me = np.asarray(d["me_xsec"])
    w = np.where(valid, me * n_target * J_2body, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    c = w / n
    omega = E_beam - k_lep[:, 0]
    ke_mag = np.linalg.norm(k_lep[:, 1:], axis=1)
    cos_th = np.clip(k_lep[:, 3] / np.clip(ke_mag, 1e-9, None), -1, 1)
    theta_deg = np.degrees(np.arccos(cos_th))
    return dict(c=c, omega=omega, theta=theta_deg, valid=valid,
                k_e=k_e, k_lep=k_lep, p_struck=p_struck, p_out=p_out,
                me=me, is_p=np.full(n, bool(is_proton)),
                had_mass=np.full(n, had_mass), n_target=np.full(n, n_target))


_SCALAR = ("c", "omega", "theta", "is_p")
_VEC = ("k_e", "k_lep", "p_struck", "p_out")
_REC_EXTRA = ("me", "had_mass", "n_target")


def generate(n, material="C", seed=0, E_beam=E_BEAM_JLAB, chunk=500_000, records=False,
             theta_acc=THETA_ACC):
    """Inclusive (e,e') MC: n TOTAL samples, split across the struck species (p, n).  Returns per-event contribution c [nb]
    (SUM = sigma), omega [MeV], theta_e' [deg], and the struck-species tag.  Chunked to bound JAX mem.
    records=True also keeps the per-event kinematics (k_e, k_lep, p_struck, p_out, me, had_mass,
    n_target) for the theta-ACCEPTED events only, the inputs needed to recompute the EM matrix
    element under a knob reweight."""
    Z, N, sf_p_path, sf_n_path = spectral_inputs(material)
    sf_p = SpectralFunction(sf_p_path); sf_n = SpectralFunction(sf_n_path)
    lo, hi = theta_acc
    keep = list(_SCALAR) + (list(_VEC) + list(_REC_EXTRA) if records else [])
    out = {k: [] for k in keep}
    species = [(True, MASS_PDG_PROTON, MASS_PDG_PROTON, sf_p, Z),
               (False, MASS_PDG_NEUTRON, MASS_PDG_NEUTRON, sf_n, N)]
    nsp = len(species)
    for si, (is_p, m_kin, m_flux, sf, n_tgt) in enumerate(species):
        n_s = n // nsp + (1 if si < n % nsp else 0)
        done = 0; sd = seed * 100 + (0 if is_p else 50)
        while done < n_s:
            m = min(chunk, n_s - done)
            rng = np.random.default_rng(sd); sd += 1
            r = _sample_species(m, rng, E_beam, m_kin, m_flux, is_p, sf, n_tgt)
            r["c"] = r["c"] * m / n_s
            sel = np.ones(m, bool) if not records else \
                (r["theta"] >= lo) & (r["theta"] <= hi) & r["valid"]
            for k in keep:
                out[k].append(r[k][sel])
            done += m
    return {k: np.concatenate(v) for k, v in out.items()}


def dsigma_domega(res, edges, theta_acc=THETA_ACC):
    """Weighted histogram of omega [MeV] inside the theta acceptance -> dsigma/domega [nb/MeV]."""
    lo, hi = theta_acc
    m = (res["theta"] >= lo) & (res["theta"] <= hi)
    h, _ = np.histogram(res["omega"][m], bins=edges, weights=res["c"][m])
    bw = np.diff(edges)
    return h / bw


from adonis.core.sample import Sampler


class EEChannel(Sampler):
    """Inclusive (e,e') QE (EM probe, monochromatic e- beam) as a detached kind-1 SAMPLER.
    `propose(seed, n)` = the verbatim `generate(..., records=True)` (theta-accepted proposal +
    per-event contribution `c`).  bit-for-bit."""

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
