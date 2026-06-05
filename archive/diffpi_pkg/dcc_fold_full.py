"""Faithful nuclear fold of the full hadron-tensor port (Phase-2.5): the differentiable
neutrino-CC dsigma/dW,dQ2 built to match ACHILLES (RES_Spectral_Func) bit-for-bit.

Per event (kind-1 reweighting; the proposal is FIXED/detached, only the L.W weight
carries the knobs -> exact gradients):

  1. sample outgoing-lepton (E', theta) and struck-nucleon (p, E_removal ~ S(p,E));
  2. struck nucleon OFF-SHELL energy  E_struck = mqe - E_removal   (res_spec_init_wgt);
  3. W^2 = (q + p_struck)^2  with q the TRUE leptonic transfer  (= ACHILLES (kpi+pp1)^2);
  4. ACHILLES current_init on-shell rebalance of the photon energy:
        q'_0 = omega - E_removal - T_N,   T_N = sqrt(p^2+mqe^2) - mqe
        Q2_adj = |q_vec|^2 - q'_0^2       (the amplitude/cut use THIS, not bare Q2);
  5. hard cuts (currents_pi_dcc.f90): W in [1076.957, 2000], Q2_adj in [0, 5e6] -> else 0;
  6. weight = (E'/E) sin(theta)            [leptonic phase space/flux, dOmega=2pi sin th dth]
            * k_pi(W)/W                    [piN 2-body phase space = fnuc k_pi/(16 pi^3 W)]
            * L_{mu,nu} W^{mu,nu}          [real CC lepton tensor x full hadron tensor].

The hadron tensor W^{mu,nu}(W, Q2_adj) is the faithful amp_dcc_sl.f port (hadron_xsec);
the lepton tensor uses the TRUE leptonic q (the true-q / adjusted-q split is ACHILLES's
off-shell prescription). Reproduces the oracle dsigma/dW to relL2 ~ 4e-4.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .dcc import DCCKnobs
from .spectral import load_spectral, SpectralSampler
from .hadron_xsec import HadronStructure
from .lepton_tensor import cm_lepton_momenta, lepton_tensor_cc, contract
from .achilles_const import MQE, M_PI, W_THR, W_MAX, Q2_MAX

E_NU_DEFAULT = 1500.0


def pion_cm_momentum(W):
    """Pion CM momentum k_pi(W) [MeV] with the exact DCC masses; 0 below threshold."""
    thr_hi = (MQE + M_PI) ** 2
    thr_lo = (MQE - M_PI) ** 2
    lam = jnp.clip((W ** 2 - thr_hi) * (W ** 2 - thr_lo), 0.0, None)
    return jnp.sqrt(lam) / (2.0 * W)


def fold_full_events(knobs: DCCKnobs, key, n=400000, hs: HadronStructure | None = None,
                     sf="pke12p_tot.data", e_nu=E_NU_DEFAULT,
                     ep_lo=50.0, ep_hi=1480.0, theta_max_deg=180.0):
    """Faithful CC fold -> (W [MeV], Q2_lep [MeV^2], weight). The returned Q2 is the
    OBSERVABLE (true leptonic Q2 = -(k-k')^2, what the oracle parse measures and what
    dsigma/dQ2 is binned in); the amplitude is internally evaluated at the on-shell
    rebalanced Q2_adj. W,Q2 detached; weight differentiable in knobs (axial_MA, pw_norm)."""
    hs = hs or HadronStructure()
    klep, ksf, kth = jax.random.split(key, 3)
    Ep = jax.lax.stop_gradient(ep_lo + (ep_hi - ep_lo) * jax.random.uniform(klep, (n,)))
    theta = jax.lax.stop_gradient(jnp.deg2rad(theta_max_deg) * jax.random.uniform(kth, (n,)))

    # leptonic transfer (true q): nu along +z, lepton in x-z plane
    omega = e_nu - Ep
    qx = -Ep * jnp.sin(theta)
    qz = e_nu - Ep * jnp.cos(theta)
    q_vec2 = qx ** 2 + qz ** 2
    Q2_lep = q_vec2 - omega ** 2                    # observable (true leptonic Q^2)
    k_lab = jnp.stack([jnp.full((n,), e_nu), jnp.zeros((n,)), jnp.zeros((n,)),
                       jnp.full((n,), e_nu)], axis=-1)
    kp_lab = jnp.stack([Ep, Ep * jnp.sin(theta), jnp.zeros((n,)), Ep * jnp.cos(theta)], axis=-1)

    # struck nucleon: off-shell energy E_struck = mqe - E_removal
    p_vec, E_rm = SpectralSampler(load_spectral(sf)).sample(ksf, n)
    p2 = jnp.sum(p_vec ** 2, axis=1)
    p_struck = jnp.concatenate([(MQE - E_rm)[:, None], p_vec], axis=1)

    # W^2 = (q + p_struck)^2 (true q); detached
    tot = jnp.stack([omega, qx, jnp.zeros((n,)), qz], axis=-1) + p_struck
    W2 = tot[:, 0] ** 2 - jnp.sum(tot[:, 1:] ** 2, axis=1)
    W = jnp.sqrt(jnp.clip(W2, 1.0, None))

    # on-shell rebalanced photon energy -> Q2_adj for the amplitude + cut
    T_N = jnp.sqrt(p2 + MQE ** 2) - MQE
    qp0 = omega - E_rm - T_N
    Q2_adj = q_vec2 - qp0 ** 2

    W = jax.lax.stop_gradient(W)
    Q2_adj = jax.lax.stop_gradient(Q2_adj)
    Q2_lep = jax.lax.stop_gradient(Q2_lep)

    # faithful ACHILLES cuts (currents_pi_dcc.f90: current = 0 outside)
    cut = (W > W_THR) & (W < W_MAX) & (Q2_adj > 0.0) & (Q2_adj < Q2_MAX)

    # lepton tensor (true q, piN-CM, q along z) x full hadron tensor at (W, Q2_adj)
    k_cm, kp_cm = cm_lepton_momenta(k_lab, kp_lab, p_struck)
    Lmn = lepton_tensor_cc(k_cm, kp_cm)
    Wmn = hs.tensor_at(W, jnp.clip(Q2_adj, 1.0, None), knobs)
    LW = contract(Lmn, Wmn)

    # weight: leptonic phase space x piN phase space x L.W, zeroed outside the cuts
    kpi = pion_cm_momentum(W)
    w = jnp.where(cut, (Ep / e_nu) * jnp.sin(theta) * (kpi / W) * LW, 0.0)
    return W, Q2_lep, w


def fold_full_dsigma_dW(knobs: DCCKnobs, W_edges, key, n=400000,
                        hs: HadronStructure | None = None, **kw):
    """Differentiable dsigma/dW histogram from the faithful fold."""
    W, Q2, w = fold_full_events(knobs, key, n, hs, **kw)
    iw = jax.lax.stop_gradient(jnp.clip(jnp.searchsorted(W_edges, W) - 1, 0, W_edges.shape[0] - 2))
    h = jax.ops.segment_sum(w, iw, num_segments=W_edges.shape[0] - 1)
    return h / jnp.diff(W_edges)
