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
