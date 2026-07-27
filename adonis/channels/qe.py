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
from adonis.channels.spectral import SpectralFunction
from adonis.channels.backend import me_cross_section, MASS_PDG_NEUTRON

_MN = C.mN
_SMIN = (M_MU + M_P) ** 2
_TWO_PI = 2 * np.pi
N_NEUTRON = 6


def _boost(p4, beta):
    """Boost 4-vector(s) p4 (...,4) by beta (...,3) (CM->lab when beta = P/E of total)."""
    b2 = np.sum(beta ** 2, axis=-1, keepdims=True)
    g = 1.0 / np.sqrt(np.clip(1 - b2, 1e-15, None))
    bp = np.sum(beta * p4[..., 1:], axis=-1, keepdims=True)
    E = (g[..., 0] * (p4[..., 0] + bp[..., 0]))
    fac = (g - 1.0) * np.where(b2 > 1e-15, bp / np.clip(b2, 1e-15, None), 0.0) + g * p4[..., :1]
    vec = p4[..., 1:] + fac * beta
    return np.concatenate([E[..., None], vec], axis=-1)


def sample(n, seed=0):
    """Flat-MC sample of the QE phase space.  Returns dict with lab momenta and event_weight J."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, 7))
    flux = SpectrumFlux()
    # ---- beam ----
    minE = flux.seed_min_GeV()
    E_GeV, J_beam = flux.sample_beam(u[:, 4], minE)        # 'is' (default) | 'flat'  -- BEAM_MODE toggle
    Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    # ---- struck nucleon (QESpectralMapper) ----
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
    det = Enu ** 2 + mom ** 2 + 2 * (pvec[:, 2] * Enu) + _SMIN          # pvec.k_nu = pz*Enu
    emax = _MN + Enu - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(emax, _MN - mom); emax = np.clip(emax, None, 400.0)
    energy = emax * u[:, 3] - 1e-8                                       # removal energy
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    J_had = mom ** 2 * dp * (cosTm + 1) * _TWO_PI * emax
    # ---- two-body final state (TwoBodyMapper), fixed-z CM axis (rotation-irrelevant) ----
    p01 = k_nu + p_struck
    s = p01[:, 0] ** 2 - np.sum(p01[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = M_MU ** 2, M_P ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * u[:, 5] - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * u[:, 6]
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = p01[:, 1:] / p01[:, 0:1]
    k_mu_cm = np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1)
    p_out_cm = np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1)
    k_mu = _boost(k_mu_cm, beta); p_out = _boost(p_out_cm, beta)
    J_2body = (cts * 0 + 2.0) * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    # validity mask
    valid = (dp > 0) & (emax > 0) & (s > _SMIN) & (lam > 0) & (radical >= 0)
    J = J_beam * J_had * J_2body
    J = np.where(valid, J, 0.0)
    return dict(k_nu=k_nu, p_struck=p_struck, k_mu=k_mu, p_out=p_out, J=J, energy=energy, mom=mom)


def generate(n, seed=0, sf=None):
    s = sample(n, seed)
    sf = sf or SpectralFunction("data/Spectral_Functions/pke12n_tot.data")
    # matrix-element factors (amps2 * flux * spinavg)
    d = me_cross_section(jnp.asarray(s["k_nu"]), jnp.asarray(s["k_mu"]),
                         jnp.asarray(s["p_struck"]), jnp.asarray(s["p_out"]),
                         spin_avg=0.5, had_mass=MASS_PDG_NEUTRON)
    me = np.asarray(d["me_xsec"])
    # initial-state weight (spectral), vectorised: removal energy = energy (= mN - E_struck)
    iw = N_NEUTRON * sf.batch(s["mom"], s["energy"])
    w = me * iw * s["J"]
    w = np.where(np.isfinite(w), w, 0.0)
    sigma = w.mean()
    return dict(sigma=sigma, w=w, k_nu=s["k_nu"], k_mu=s["k_mu"], p_struck=s["p_struck"],
                p_out=s["p_out"])


def sample_importance(n, seed=0, sf=None, n_neutron=N_NEUTRON):
    """Like sample() but draws the struck nucleon from the spectral IMPORTANCE sampler
    (|p|,E ~ |p|^2 S) instead of the flat QESpectralMapper -> the dominant flat-MC weight
    variance (the |p|^2 S peak) cancels in the weight.  event.Weight() loses J_had and initwgt
    (now in the sampling); they are replaced by the constant N_neutron (the # of target NEUTRONS,
    A-Z; CC QE is nu n->mu- p).  Default 6 = carbon; pass the target's A-Z for other nuclei.
    Beam + TwoBody final state sampled as in sample()."""
    from adonis.channels.spectral import SpectralImportanceSampler, SpectralFunction as _SF
    rng = np.random.default_rng(seed)
    u = rng.random((n, 7))
    flux = SpectrumFlux()
    minE = flux.seed_min_GeV()
    E_GeV, J_beam = flux.sample_beam(u[:, 4], minE); Enu = E_GeV * 1000.0   # BEAM_MODE toggle
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    sf = sf or _SF("data/Spectral_Functions/pke12n_tot.data")
    samp = SpectralImportanceSampler(sf)
    pvec, E_rm = samp.sample(n, rng)
    p_struck = np.concatenate([(_MN - E_rm)[:, None], pvec], axis=1)
    # ThreeBody->TwoBody: total -> mu + p, isotropic CM (same as sample())
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = M_MU ** 2, M_P ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * u[:, 5] - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = _TWO_PI * u[:, 6]
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    k_mu = _boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1), beta)
    p_out = _boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1), beta)
    # Kinematically-degenerate events (s->0 => CM boost beta->1) blow the boost up to ~1e17 MeV.
    # They are rejected below (valid=False => w=0), but the garbage momenta are toxic to any downstream
    # consumer: the FSI cascade (absurd momenta sit on escape/Pauli thresholds -> CPU/GPU divergence)
    # AND me_cross_section (degenerate kinematics -> NaN amps2).  Replace the bad rows' FULL kinematics
    # with a valid event's, so every consumer sees finite, self-consistent QE kinematics.  w=0 (set
    # below via `valid`, which is computed from the ORIGINAL per-row s/lam/E_rm) keeps physics identical.
    _bad = (~np.isfinite(p_out).all(1) | ~np.isfinite(k_mu).all(1)
            | (np.linalg.norm(p_out[:, 1:], axis=1) > 1.0e6)
            | (np.linalg.norm(k_mu[:, 1:], axis=1) > 1.0e6))
    if _bad.any() and not _bad.all():
        _ref = int(np.argmin(_bad))                     # first finite row
        for _a in (k_nu, k_mu, p_struck, p_out):
            _a[_bad] = _a[_ref]
    J_2body = 2.0 * _TWO_PI * pcm / (sqrts * 16 * np.pi ** 2)
    # ACHILLES QESpectralMapper restricts the removal energy to [0, emax]; the importance sampler
    # draws E_rm from the full spectral grid, so the SAME kinematic ceiling must be imposed or the
    # high-|p|/high-E tail (-> high delta_pT) is over-populated.  emax = min(mN+E0-sqrt(det),
    # mN-mom, 400), det = E0^2 + mom^2 + 2 (p.k_nu) + Smin  (HadronicMapper.cc:50-53).
    mom_s = np.linalg.norm(pvec, axis=1)
    det_e = Enu ** 2 + mom_s ** 2 + 2 * pvec[:, 2] * Enu + (M_MU + M_P) ** 2
    emax = _MN + Enu - np.sqrt(np.clip(det_e, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom_s), 400.0)
    # Lower removal-energy bound = the SF's OWN grid start (NOT a hardcoded 2.5, which was the carbon
    # pke12 grid start: a no-op for C but for Ar (pke40 grid from 0) it wrongly discarded ~0.7% of the
    # spectral strength in [0, 2.5] MeV that ACHILLES keeps -- ACHILLES samples E_rm from 0 with no floor
    # (HadronicMapper.cc), the SF's own 0-outside-grid does the flooring).  Fixes the ~1% Ar QE deficit.
    valid = (s > (M_MU + M_P) ** 2) & (lam > 0) & (E_rm > sf.energy[0]) & (E_rm < emax)
    d = me_cross_section(jnp.asarray(k_nu), jnp.asarray(k_mu), jnp.asarray(p_struck),
                         jnp.asarray(p_out), spin_avg=0.5, had_mass=MASS_PDG_NEUTRON)
    me = np.asarray(d["me_xsec"])
    w = np.where(valid, me * n_neutron * J_2body * J_beam, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    return dict(w=w, k_nu=k_nu, k_mu=k_mu, p_struck=p_struck, p_out=p_out, sigma=w.mean())


from adonis.core.sample import Sampler       # noqa: E402


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
