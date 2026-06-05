"""Total single-pion cross section vs neutrino energy, σ(E_ν) (Phase A3).

Integrates the DCC single-pion channel's per-event weight over the sampled lepton
phase space at a fixed beam energy, for the three CC isospin channels
(ch0 n→pπ⁰, ch1 n→nπ⁺, ch2 p→pπ⁺).  Built for the free-nucleon target
(`adonis.nuclear.free.FreeNucleon`) → the paper's ANL/BNL comparison (Fig anl_bnl),
but works for any NuclearModel.

Estimator.  The channel samples the outgoing lepton (E′ uniform in [ep_lo, ep_hi],
θ uniform in [0, θ_max]) and the pion solid angle (uniform over 4π, already folded
into the weight as the explicit 4π factor).  So the Monte-Carlo estimate of the
total cross section is

    σ = V_lep · ⟨ w ⟩ ,   V_lep = (ep_hi − ep_lo) · θ_max ,

with ⟨·⟩ the sample mean (which also averages the pion-angle integral).  Per channel,
w_c = 1(cut) · prefac · 4π · mult_c · (L·W)_c, and Σ_c w_c = w, so Σ_c σ_c = σ_total.

These σ are in the model's internal (relative) units — the universal absolute
constant G_F²cos²θ_c/(flux·measure) is one energy/channel-independent factor applied
in A3.5.  The **energy dependence** (shape) and **channel ratios** are parameter-free
predictions independent of that constant, and are what Fig 2 tests.

Differentiability.  σ_c(knobs) is differentiable in the physics knobs (e.g. axial_MA)
exactly as the per-event weight is — the proposal is fixed/detached, only the weight
carries the knobs (kind-1 reweighting).  `dsigma_dMA_closure` gates dσ/dM_A.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp

from adonis.params import GenConfig, DCCKnobs
from adonis.primary.dcc.channel import DCCSinglePion
from adonis.nuclear.free import FreeNucleon

# Calibrated model→nb scale.  The relative weight omits the universal CC prefactor
# (G_F²cos²θ_c and the proposal-measure/flux bookkeeping); this ONE constant — fit against
# the ACHILLES free-nucleon oracle, which reports nb (Constants.hh HBARC2=0.38938 mb·GeV²) —
# converts the model's relative σ to physical nb.  It is unit bookkeeping, NOT a physics
# tune: energy- and channel-independent, and the SAME for ν_e and ν_μ to 0.2% (see
# `freenucleon_sigma_oracle`, which re-fits it each call rather than relying on this value).
SIGMA_UNIT_NB = 2.569e-14        # nb per (model relative unit); ~0.2% calibration spread
CM2_1E38_PER_NB = 1.0e5          # 1 nb = 1e5 × 10⁻³⁸ cm²  (the Fig-2 σ axis)


def _channel_for(e_nu, cfg, nuclear, m_lep):
    """A DCCSinglePion configured for a stationary-target σ(E_ν) point: lepton energy
    proposal spans [m_lep, e_nu] (the kinematic range for a nucleon at rest)."""
    ep_lo = float(m_lep)
    ep_hi = float(e_nu)
    c = replace(cfg, e_nu=float(e_nu), ep_lo=ep_lo, ep_hi=ep_hi, m_lep=float(m_lep))
    return DCCSinglePion(c, flux=None, nuclear=nuclear), ep_lo, ep_hi


def _channel_sigma_terms(ch, knobs, S, V):
    """Per-channel weight wc (N, n_ch) with Σ_c wc = w, scaled so ⟨wc⟩·1 gives σ_c when
    averaged over the sample (the proposal volume V folds in at the mean)."""
    w, LWc = ch.weight(knobs, S)                        # w (N,), LWc (N, n_ch)
    # per-channel weight: 1(cut)·prefac·4π·mult_c·(L·W)_c ; Σ_c = w
    return jnp.where(S["cut"][:, None], 1.0, 0.0) * (
        S["prefac"][:, None] * (4.0 * jnp.pi) * S["mult"][None, :] * LWc) * V


def sigma_channels_at(knobs, key, e_nu, n, cfg=GenConfig(spline=False),
                      nuclear=None, m_lep=0.0, chunk=None):
    """Per-channel σ (n_ch,) and total σ at a single beam energy `e_nu` [MeV].

    Returns (sigma_total, sigma_per_channel[n_ch]) in the model's relative units.
    Differentiable in `knobs`.  `chunk` bounds peak memory (the per-event angular
    kernels are large): the estimator is a plain mean over events, so chunks just
    accumulate the running sum and count."""
    nuclear = FreeNucleon() if nuclear is None else nuclear
    ch, ep_lo, ep_hi = _channel_for(e_nu, cfg, nuclear, m_lep)
    V = (ep_hi - ep_lo) * np.deg2rad(cfg.theta_max_deg)
    if not chunk or chunk >= n:
        S = ch.sample(key, n)
        sigma_c = jnp.mean(_channel_sigma_terms(ch, knobs, S, V), axis=0)
        return jnp.sum(sigma_c), sigma_c
    acc = None
    done = 0
    i = 0
    while done < n:
        m = min(chunk, n - done)
        S = ch.sample(jax.random.fold_in(key, i), m)
        s = jnp.sum(_channel_sigma_terms(ch, knobs, S, V), axis=0)
        acc = s if acc is None else acc + s
        done += m; i += 1
    sigma_c = acc / done
    return jnp.sum(sigma_c), sigma_c


def sigma_vs_enu(knobs, key, energies, n=60_000, cfg=GenConfig(spline=False),
                 nuclear=None, m_lep=0.0):
    """σ(E_ν) scan. `energies` [MeV] iterable. Returns
    (sigma_total[E], sigma_per_channel[E, n_ch]) as numpy arrays (relative units)."""
    nuclear = FreeNucleon() if nuclear is None else nuclear
    tot, perch = [], []
    for i, e in enumerate(energies):
        st, sc = sigma_channels_at(knobs, jax.random.fold_in(key, i), e, n,
                                   cfg=cfg, nuclear=nuclear, m_lep=m_lep)
        tot.append(float(st)); perch.append(np.asarray(sc))
    return np.array(tot), np.stack(perch, axis=0)


def freenucleon_sigma_oracle(csv=None, key=None, n=120_000, knobs=None,
                             rel_max=0.06, std_max=0.03, chunk=25_000, m_lep=0.0):
    """Oracle gate (A3.4): the free-nucleon σ(E_ν) for all three CC isospin channels
    vs ACHILLES on stationary nucleons (1H proton + 1N free neutron).

    The model is in relative units; ACHILLES in its own internal units.  A SINGLE
    universal constant c (energy- and channel-independent) bridges them, so the test
    fits one c (least squares, log space) over the whole 9-energy × 3-channel grid and
    asserts (a) the residual |c·model − ACH|/ACH is small everywhere and (b) the spread
    of the per-cell c is tight — i.e. the **energy dependence AND the channel ratios**
    are reproduced with one normalisation.  Returns a TestResult; metrics include the
    fitted constant `c` (model→ACHILLES units)."""
    from pathlib import Path
    from adonis.core.validation import TestResult
    key = jax.random.PRNGKey(11) if key is None else key
    knobs = DCCKnobs() if knobs is None else knobs
    if csv is None:
        csv = Path(__file__).resolve().parents[3] / "data" / "oracle" / "freenucleon_nue_sigma.csv"
    ref = np.loadtxt(csv)
    E, ach = ref[:, 0], ref[:, 1:4]                    # ach (nE, 3): ch0,ch1,ch2
    mod = np.zeros_like(ach)
    for i, e in enumerate(E):
        _, sc = sigma_channels_at(knobs, jax.random.fold_in(key, i), float(e), n,
                                  chunk=chunk, m_lep=m_lep)
        mod[i] = np.asarray(sc)
    # single bridging constant in log space (geometric-mean of ACH/model)
    c = float(np.exp(np.mean(np.log(ach / mod))))
    cells = ach / mod / c                              # want ~1 in every cell
    rel = np.abs(c * mod - ach) / ach
    passed = bool(rel.max() < rel_max and cells.std() < std_max)
    return TestResult(
        "FreeNucleon.sigma_enu.oracle", "oracle", passed, False,
        f"σ(E_ν) 3-channel vs ACHILLES: max rel {rel.max():.3f} mean {rel.mean():.3f}, "
        f"c-spread std {cells.std():.3f} (tol rel<{rel_max}, std<{std_max})",
        {"rel_max": float(rel.max()), "rel_mean": float(rel.mean()),
         "c_spread_std": float(cells.std()), "c": c, "n_cells": int(ach.size)})


def sigma_vs_enu_nb(knobs, key, energies, n=60_000, cfg=GenConfig(spline=False),
                    nuclear=None, m_lep=0.0):
    """σ(E_ν) in **physical nb** (= relative σ × SIGMA_UNIT_NB). Returns
    (sigma_total[E], sigma_per_channel[E, n_ch]). Multiply by `CM2_1E38_PER_NB` for
    10⁻³⁸ cm² (the paper's Fig-2 units)."""
    tot, perch = sigma_vs_enu(knobs, key, energies, n=n, cfg=cfg, nuclear=nuclear, m_lep=m_lep)
    return tot * SIGMA_UNIT_NB, perch * SIGMA_UNIT_NB


# --------------------------------------------------------------------------- #
#  EM (electron probe) single-pion -- Phase A1
# --------------------------------------------------------------------------- #
_EM_HS = {}


def _em_hs(spline=False):
    """Cached EM HadronStructure (channels=EM_CHANNELS, mode=10). Building the angular
    kernels is expensive, so reuse across energies."""
    from adonis.primary.dcc.structure import HadronStructure, EM_CHANNELS
    key = spline
    if key not in _EM_HS:
        _EM_HS[key] = HadronStructure(channels=EM_CHANNELS, n_theta=12, n_phi=12, spline=spline)
    return _EM_HS[key]


def em_sigma_channels_at(knobs, key, e_e, n, theta_min_deg=10.0, theta_max_deg=90.0,
                         nuclear=None, chunk=None):
    """Per-channel EM single-pion cross section within the lepton angular acceptance
    [theta_min, theta_max] at electron energy `e_e` [MeV], for the four EM channels
    (p->p pi0, p->n pi+, n->n pi0, n->p pi-).  Reuses the validated sample/weight machinery
    with current='EM' (lepton_tensor_em x 1/Q^4) and EM_CHANNELS.  Returns
    (sigma_total, sigma_per_channel[4]) in relative units; differentiable in `knobs`.

    Unlike the CC total, the EM total is 1/Q^4-divergent at forward angles, so a finite
    angular acceptance (theta_min>0) is REQUIRED (matches ACHILLES's AngleTheta cut)."""
    from adonis.primary.dcc.channel import sample_final_state, weight_from_sample
    nuclear = FreeNucleon() if nuclear is None else nuclear
    hs = _em_hs()
    V = (float(e_e) - 0.0) * (np.deg2rad(theta_max_deg) - np.deg2rad(theta_min_deg))

    def terms(k, m):
        S = sample_final_state(k, m, hs=hs, e_nu=float(e_e), ep_lo=0.0, ep_hi=float(e_e),
                               theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg,
                               nuclear=nuclear, m_lep=0.0, current="EM")
        _, LWc = weight_from_sample(knobs, S, use_spline=False)
        wc = jnp.where(S["cut"][:, None], 1.0, 0.0) * (
            S["prefac"][:, None] * (4.0 * jnp.pi) * S["mult"][None, :] * LWc) * V
        return wc

    if not chunk or chunk >= n:
        sigma_c = jnp.mean(terms(key, n), axis=0)
        return jnp.sum(sigma_c), sigma_c
    acc, done, i = None, 0, 0
    while done < n:
        m = min(chunk, n - done)
        s = jnp.sum(terms(jax.random.fold_in(key, i), m), axis=0)
        acc = s if acc is None else acc + s
        done += m; i += 1
    sigma_c = acc / done
    return jnp.sum(sigma_c), sigma_c


def em_dsigma_dpw_closure(key=None, e_e=1500.0, n=30_000, pw_index=5, eps=2e-3, tol=1e-3):
    """EM closure (A1): d(total EM σ)/d(vector-FF knob), autodiff vs central FD.

    EM has no axial current, so the differentiable handle is the per-partial-wave
    normalisation `pw_norm` (here on the P33/Δ wave, index 5) -- the EM analog of A3's
    dσ/dM_A. The proposal is sampled once and reweighted, so the gradient is exact."""
    from adonis.core.validation import grad_closure
    key = jax.random.PRNGKey(0) if key is None else key

    def total_em(scale):
        pw = tuple(jnp.where(jnp.arange(14) == pw_index, scale, 0.0))
        tot, _ = em_sigma_channels_at(DCCKnobs(pw_norm=pw), key, e_e, n)
        return tot

    return grad_closure(total_em, "EM.sigma.closure", x0=0.0, eps=eps, tol=tol)


def em_sigma_oracle(csv=None, key=None, n=120_000, knobs=None, rel_max=0.06,
                    std_max=0.03, chunk=25_000, theta_min_deg=10.0, theta_max_deg=90.0):
    """EM oracle gate (A1): EM single-pion σ within the [theta_min,theta_max] acceptance
    for the four channels vs ACHILLES (electron on 1H proton channels + 1N neutron
    channels, AngleTheta cut). Same single-universal-constant bridge as the CC oracle.

    CSV columns: E_e[MeV], ch0 (p->p pi0), ch1 (p->n pi+), ch2 (n->n pi0), ch3 (n->p pi-).
    Gates all FOUR channels (the neutron pi0/pi- split is now correct -- the isign=-1
    neutron-amplitude phase, amp_dcc_sl_module.f:644)."""
    from pathlib import Path
    from adonis.core.validation import TestResult
    key = jax.random.PRNGKey(7) if key is None else key
    knobs = DCCKnobs() if knobs is None else knobs
    if csv is None:
        csv = Path(__file__).resolve().parents[3] / "data" / "oracle" / "freenucleon_em_sigma.csv"
    ref = np.loadtxt(csv)
    E, ach = ref[:, 0], ref[:, 1:5]
    mod = np.zeros_like(ach)
    for i, e in enumerate(E):
        _, sc = em_sigma_channels_at(knobs, jax.random.fold_in(key, i), float(e), n,
                                     theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg,
                                     chunk=chunk)
        mod[i] = np.asarray(sc)
    c = float(np.exp(np.mean(np.log(ach / mod))))
    cells = ach / mod / c
    rel = np.abs(c * mod - ach) / ach
    passed = bool(rel.max() < rel_max and cells.std() < std_max)
    return TestResult(
        "EM.sigma.oracle", "oracle", passed, False,
        f"EM σ 4-channel vs ACHILLES: max rel {rel.max():.3f} "
        f"mean {rel.mean():.3f}, c-spread std {cells.std():.3f} (tol rel<{rel_max}, std<{std_max})",
        {"rel_max": float(rel.max()), "rel_mean": float(rel.mean()),
         "c_spread_std": float(cells.std()), "c": c, "n_cells": int(ach.size)})


def dsigma_dMA_closure(key=None, e_nu=1500.0, n=40_000, eps=2e-3, tol=1e-3,
                       cfg=GenConfig(spline=False), nuclear=None, m_lep=0.0):
    """Closure: d(total σ)/dM_A, autodiff vs central finite difference at fixed key.

    The proposal is sampled ONCE and reweighted at M_A ± eps, so this is the exact
    kind-1 gradient gate for the free-nucleon σ(E_ν)."""
    from adonis.core.validation import grad_closure
    key = jax.random.PRNGKey(0) if key is None else key
    nuclear = FreeNucleon() if nuclear is None else nuclear
    ch, _, _ = _channel_for(e_nu, cfg, nuclear, m_lep)
    S = ch.sample(key, n)

    def total_sigma(MA):
        w, _ = ch.weight(DCCKnobs(axial_MA=MA), S)
        return jnp.sum(w)

    return grad_closure(total_sigma, "FreeNucleon.sigma_enu.closure",
                        x0=1.0, eps=eps, tol=tol)
