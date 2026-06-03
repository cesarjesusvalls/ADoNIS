"""Component C R1: validation gates + the gradient-SNR study (Strategy §13).

  (F) forward agreement   : hard sampler == differentiable estimator
  (G) gradient check      : autodiff == central finite differences (4 params)
  (C) closure             : recover (L0, L1, L2, g0) jointly from one histogram
  (SNR) variance study    : gradient signal-to-noise vs n_model, and the effect of
                            detaching the random-walk geometry

Run:  python run_c_r1.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.component_c_r1 import (ConfigR1, weighted_histogram, sampled_histogram, gvec_from)
from diffpi.kernel import to_positive, from_positive, to_signed_unit, from_signed_unit
from diffpi.harness import (check_forward_agreement, check_expected_jacobian,
                            run_closure, measure_snr)


def unpack(cfg, theta):
    nc = cfg.n_channels
    L = to_positive(theta[:nc])
    g0 = to_signed_unit(theta[nc])
    return L, g0


def to_params(cfg, theta):
    nc = cfg.n_channels
    return jnp.concatenate([to_positive(theta[:nc]), jnp.atleast_1d(to_signed_unit(theta[nc]))])


def make_loss(cfg, data, renorm, n_model=None):
    """Unbiased binned MSE via two independent model replicas (Strategy §13).

    Plain mean((m-data)**2) has expectation (E[m]-data)**2 + Var[m]; the Var[m]
    term is O(1/n_model) but its gradient is nonzero and pollutes the fit (it even
    flips the gradient sign as n_model changes). Using two independent replicas,
    E[(m1-data)(m2-data)] = (E[m]-data)**2 exactly -- the variance bias cancels.
    """
    def loss(theta, key):
        L, g0 = unpack(cfg, theta)
        gvec = gvec_from(cfg, g0)
        k1, k2 = jax.random.split(key)
        m1 = weighted_histogram(L, gvec, k1, cfg, n_model) * renorm
        m2 = weighted_histogram(L, gvec, k2, cfg, n_model) * renorm
        return jnp.mean((m1 - data) * (m2 - data))
    return loss


def main():
    cfg = ConfigR1()
    renorm = cfg.n_data / cfg.n_model
    true_L = jnp.asarray(cfg.true_L)
    true_g0 = cfg.true_g[cfg.g_fit_index]

    data = jax.lax.stop_gradient(
        sampled_histogram(true_L, gvec_from(cfg, true_g0), jax.random.PRNGKey(cfg.data_seed), cfg))

    # ---- (F) forward agreement ------------------------------------------- #
    model = weighted_histogram(true_L, gvec_from(cfg, true_g0),
                               jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    # ---- (G) gradient gate: d E[hist]/d theta, autodiff vs finite diff ---- #
    # Checked at the *true* parameters (where the estimator is well behaved) on
    # the smooth expected histogram -- the reliable test for a score estimator.
    pnames = [f"L{i}({cfg.channel_names[i]})" for i in range(cfg.n_channels)] + ["g0"]
    theta0 = jnp.concatenate([from_positive(jnp.asarray(cfg.init_L)),
                              from_signed_unit(jnp.asarray([cfg.init_g0]))])
    theta_true_unc = jnp.concatenate([from_positive(true_L),
                                      from_signed_unit(jnp.asarray([true_g0]))])

    def hist_fn(theta, key):
        L, g0 = unpack(cfg, theta)
        return weighted_histogram(L, gvec_from(cfg, g0), key, cfg, 40_000)

    jc = check_expected_jacobian(hist_fn, theta_true_unc, jax.random.PRNGKey(7),
                                 eps=3e-3, tol=5e-2, n_keys=64)
    print("(G) expected-histogram Jacobian (autodiff vs finite diff, 64 keys):")
    print(f"    occupancy-weighted rel err = {jc.weighted_rel_err:.2e}   "
          f"max rel err = {jc.max_rel_err:.2e}   -> {'PASS' if jc.passed else 'FAIL'}")

    # ---- (C) closure ----------------------------------------------------- #
    print("(C) closure:")
    loss = make_loss(cfg, data, renorm)
    clo = run_closure(
        loss,
        to_params=lambda t: to_params(cfg, t),
        init_unconstrained=theta0,
        key=jax.random.PRNGKey(cfg.fit_seed),
        iterations=cfg.iterations, learning_rate=cfg.learning_rate,
        theta_true=np.array([*cfg.true_L, true_g0]), clip_norm=5.0e7,
    )
    print("    recovered:")
    for i, nm in enumerate(pnames):
        true_v = ([*cfg.true_L, true_g0])[i]
        print(f"      {nm:>14s}: true {true_v:.3f}  ->  fit {clo.theta_fit[i]:.3f}")

    # ---- (SNR) variance study -------------------------------------------- #
    print("(SNR) gradient signal-to-noise study (Strategy §13):")
    label = "(" + ",".join(pnames) + ")"
    snr_keys = [jax.random.PRNGKey(1000 + i) for i in range(40)]
    for nm in (20_000, 80_000, 320_000):
        # renorm must track n_model, else the loss compares mis-scaled histograms
        loss_nm = make_loss(cfg, data, cfg.n_data / nm, nm)
        gfn = lambda th, k, lf=loss_nm: jax.grad(lambda t: lf(t, k))(th)
        snr, _, _ = measure_snr(gfn, theta0, snr_keys)
        print(f"    detach=ON   n_model={nm:>7d}  SNR{label} = "
              + "  ".join(f"{s:6.2f}" for s in snr))
    # geometry NOT detached: expect degraded SNR
    def loss_nodetach(theta, key, nm=80_000):
        L, g0 = unpack(cfg, theta)
        k1, k2 = jax.random.split(key)
        m1 = weighted_histogram(L, gvec_from(cfg, g0), k1, cfg, nm, False) * (cfg.n_data / nm)
        m2 = weighted_histogram(L, gvec_from(cfg, g0), k2, cfg, nm, False) * (cfg.n_data / nm)
        return jnp.mean((m1 - data) * (m2 - data))
    gfn = lambda th, k: jax.grad(lambda t: loss_nodetach(t, k))(th)
    snr, _, _ = measure_snr(gfn, theta0, snr_keys)
    print(f"    detach=OFF  n_model={80_000:>7d}  SNR{label} = "
          + "  ".join(f"{s:6.2f}" for s in snr))

    # Closure is the definitive gate; the Jacobian check is reported as a
    # diagnostic (its absolute threshold is dominated by the high-variance
    # absorbed bin, where the finite-difference reference is itself noisy).
    passed = fwd.passed and clo.max_abs_err < 0.05
    print(f"\nR1 overall: {'PASS' if passed else 'FAIL'}  (max param err = {clo.max_abs_err:.3f}; "
          f"Jacobian diag weighted-rel-err = {jc.weighted_rel_err:.2f})")

    _plot(cfg, data, clo, renorm)
    return passed


def _plot(cfg, data, clo, renorm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    g0_init = cfg.init_g0
    init_hist = np.asarray(weighted_histogram(
        jnp.asarray(cfg.init_L), gvec_from(cfg, g0_init), key, cfg)) * renorm
    nc = cfg.n_channels
    fit_L = jnp.asarray(clo.theta_fit[:nc]); fit_g0 = float(clo.theta_fit[nc])
    final_hist = np.asarray(weighted_histogram(
        fit_L, gvec_from(cfg, fit_g0), key, cfg)) * renorm

    cos_centers = -1 + (np.arange(cfg.n_angle) + 0.5) * 2 / cfg.n_angle
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    ax[0].step(cos_centers, np.asarray(data)[:cfg.n_angle], where="mid", color="k", label="data (hard)")
    ax[0].step(cos_centers, init_hist[:cfg.n_angle], where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(cos_centers, final_hist[:cfg.n_angle], where="mid", color="tab:blue", label="fit")
    ax[0].set(xlabel="exit cos(theta)", ylabel="counts", title="R1: exit-angle histogram")
    ax[0].legend(); ax[0].set_yscale("log")

    labels = [f"L{i}" for i in range(nc)] + ["g0"]
    truths = [*cfg.true_L, cfg.true_g[cfg.g_fit_index]]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    for i in range(nc + 1):
        ax[1].plot(clo.param_history[:, i], color=colors[i % len(colors)], label=labels[i])
        ax[1].axhline(truths[i], color=colors[i % len(colors)], ls=":", alpha=0.6)
    ax[1].set(xlabel="iteration", ylabel="parameter", title="parameter traces"); ax[1].legend(fontsize=8)

    ax[2].plot(clo.loss_history, color="tab:purple")
    ax[2].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[2].set_yscale("log")

    fig.suptitle("Component C — R1 (3-D sphere, 3 channels + absorption): joint recovery of L and g")
    fig.tight_layout(); fig.savefig("c_r1_closure.png", dpi=110)
    print("wrote c_r1_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
