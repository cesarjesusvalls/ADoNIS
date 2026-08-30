"""Absolute ACHILLES CC QE cross section + differential, by flat-MC over the unit hypercube using
the ported mappers (BeamMapper/QESpectralMapper/TwoBodyMapper) and the bit-exact integrand
amps2 * FluxFactor * InitialStateWeight * SpinAvg * event.Weight().

sigma = E_u[ integrand ] over u ~ U[0,1]^7 (4 nucleon + 1 beam + 2 final), reproducing ACHILLES's
MultiChannel MC estimate (Vegas only reduces variance; flat sampling -> same integral).  Returns
the sampled lab momenta + per-event weight so any QE differential (Fig 1, Fig 7 QE component) is a
weighted histogram on the exact ACHILLES absolute scale.  nu_mu + 12C (6 neutrons), T2K flux.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.channels import constants as C
from adonis.flux.spectrum import SpectrumFlux, M_MU, M_P
from adonis.nuclear.spectral import SpectralFunction
from adonis.channels.currents.matrix_element import me_cross_section, MASS_PDG_NEUTRON

_MN = C.mN
_SMIN = (M_MU + M_P) ** 2
_TWO_PI = 2 * np.pi
N_NEUTRON = 6


from adonis.channels.twobody import isotropic_two_body_cm


def sample(n, seed=0):
    """Flat-MC sample of the QE phase space.  Returns dict with lab momenta and event_weight J."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, 7))
    flux = SpectrumFlux()
    minE = flux.seed_min_GeV()
    E_GeV, J_beam = flux.sample_beam(u[:, 4], minE)
    Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    radical = Enu ** 2 + 2 * Enu * _MN + _MN ** 2 - _SMIN
    radical = np.clip(radical, 0, None)
    pmin = np.clip(Enu - np.sqrt(radical), 0, None)
    pmax = np.clip(Enu + np.sqrt(radical), None, 800.0)
    dp = pmax - pmin
    mom = dp * u[:, 0] + pmin
    cosTm = (2 * Enu * _MN + _MN ** 2 - mom ** 2 - _SMIN) / (2 * Enu * np.clip(mom, 1e-9, None))
    cosTm = np.clip(cosTm, -1, 1)
    cosT = (cosTm + 1) * u[:, 1] - 1; sinT = np.sqrt(np.clip(1 - cosT ** 2, 0, None))
    phi = _TWO_PI * u[:, 2]
    pvec = np.stack([mom * sinT * np.cos(phi), mom * sinT * np.sin(phi), mom * cosT], axis=1)
    det = Enu ** 2 + mom ** 2 + 2 * (pvec[:, 2] * Enu) + _SMIN
    emax = _MN + Enu - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(emax, _MN - mom); emax = np.clip(emax, None, 400.0)
    energy = emax * u[:, 3] - 1e-8
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    J_had = mom ** 2 * dp * (cosTm + 1) * _TWO_PI * emax
    p01 = k_nu + p_struck
    k_lep, p_out, pcm, sqrts, s, lam = isotropic_two_body_cm(p01, M_MU, M_P, u[:, 5], u[:, 6])
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    valid = (dp > 0) & (emax > 0) & (s > _SMIN) & (lam > 0) & (radical >= 0)
    J = J_beam * J_had * J_2body
    J = np.where(valid, J, 0.0)
    return dict(k_nu=k_nu, p_struck=p_struck, k_lep=k_lep, p_out=p_out, J=J, energy=energy, mom=mom)


def generate(n, seed=0, sf=None):
    s = sample(n, seed)
    sf = sf or SpectralFunction("data/Spectral_Functions/pke12n_tot.data")
    d = me_cross_section(jnp.asarray(s["k_nu"]), jnp.asarray(s["k_lep"]),
                         jnp.asarray(s["p_struck"]), jnp.asarray(s["p_out"]),
                         spin_avg=0.5, had_mass=MASS_PDG_NEUTRON)
    me = np.asarray(d["me_xsec"])
    iw = N_NEUTRON * sf.batch(s["mom"], s["energy"])
    w = me * iw * s["J"]
    w = np.where(np.isfinite(w), w, 0.0)
    sigma = w.mean()
    return dict(sigma=sigma, w=w, k_nu=s["k_nu"], k_lep=s["k_lep"], p_struck=s["p_struck"],
                p_out=s["p_out"])


def sample_importance(n, seed=0, sf=None, n_neutron=N_NEUTRON):
    """Like sample() but draws the struck nucleon from the spectral IMPORTANCE sampler
    (|p|,E ~ |p|^2 S) instead of the flat QESpectralMapper -> the dominant flat-MC weight
    variance (the |p|^2 S peak) cancels in the weight.  event.Weight() loses J_had and initwgt
    (now in the sampling); they are replaced by the constant N_neutron (the # of target NEUTRONS,
    A-Z; CC QE is nu n->mu- p).  Default 6 = carbon; pass the target's A-Z for other nuclei.
    Beam + TwoBody final state sampled as in sample()."""
    from adonis.nuclear.spectral import SpectralImportanceSampler, SpectralFunction as _SF
    rng = np.random.default_rng(seed)
    u = rng.random((n, 7))
    flux = SpectrumFlux()
    minE = flux.seed_min_GeV()
    E_GeV, J_beam = flux.sample_beam(u[:, 4], minE); Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    sf = sf or _SF("data/Spectral_Functions/pke12n_tot.data")
    samp = SpectralImportanceSampler(sf)
    pvec, E_rm = samp.sample(n, rng)
    p_struck = np.concatenate([(_MN - E_rm)[:, None], pvec], axis=1)
    P = k_nu + p_struck
    k_lep, p_out, pcm, sqrts, s, lam = isotropic_two_body_cm(P, M_MU, M_P, u[:, 5], u[:, 6])
    k_lep = np.array(k_lep); p_out = np.array(p_out)
    _bad = (~np.isfinite(p_out).all(1) | ~np.isfinite(k_lep).all(1)
            | (np.linalg.norm(p_out[:, 1:], axis=1) > 1.0e6)
            | (np.linalg.norm(k_lep[:, 1:], axis=1) > 1.0e6))
    if _bad.any() and not _bad.all():
        _ref = int(np.argmin(_bad))
        for _a in (k_nu, k_lep, p_struck, p_out):
            _a[_bad] = _a[_ref]
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    mom_s = np.linalg.norm(pvec, axis=1)
    det_e = Enu ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * Enu + (M_MU + M_P) ** 2
    emax = _MN + Enu - np.sqrt(np.clip(det_e, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom_s), 400.0)
    valid = (s > (M_MU + M_P) ** 2) & (lam > 0) & (E_rm > sf.energy[0]) & (E_rm < emax)
    d = me_cross_section(jnp.asarray(k_nu), jnp.asarray(k_lep), jnp.asarray(p_struck),
                         jnp.asarray(p_out), spin_avg=0.5, had_mass=MASS_PDG_NEUTRON)
    me = np.asarray(d["me_xsec"])
    w = np.where(valid, me * n_neutron * J_2body * J_beam, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    return dict(w=w, k_nu=k_nu, k_lep=k_lep, p_struck=p_struck, p_out=p_out, sigma=w.mean())


from adonis.core.sample import Sampler


class QEChannel(Sampler):
    """CC quasi-elastic (nu n -> mu- p) as a detached kind-1 SAMPLER.

    `propose(seed, n)` draws the frozen proposal -- spectral-importance struck nucleon + isotropic
    two-body (mu, p) -- carrying the NOMINAL per-event weight `w`.  This is the production kind-1
    split's "sample" half: the knob-differentiable WEIGHT is applied downstream by the production
    reweight (adonis.reweight.bank_reweight.bank_weight over the banked records), NOT re-derived here.
    The numerical kernel is the verbatim `sample_importance` function above -- bit-for-bit identical."""

    def __init__(self, sf=None, n_neutron=N_NEUTRON):
        self.sf = sf
        self.n_neutron = n_neutron

    def propose(self, key, n):
        return sample_importance(n, seed=int(key), sf=self.sf, n_neutron=self.n_neutron)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    r = generate(n)
    nz = (r["w"] > 0).sum()
    print(f"QE sigma = {r['sigma']:.4e} nb   (N={n}, nonzero {nz})   "
          f"ACHILLES target ~4.256e-5 nb")
