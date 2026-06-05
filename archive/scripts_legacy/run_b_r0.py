"""Component B R0: meson-baryon vertex (Delta in elastic pi-N), validation + closure.

  (F) forward agreement   : hard sampler == differentiable estimator
  (G) expected-histogram Jacobian : autodiff == finite difference (m_R, Gamma, b)
  (C) closure             : recover (m_R, Gamma, b) from the 2-D (W, cos theta) histogram

Run:  python run_b_r0.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.component_b_r0 import (ConfigBR0, weighted_histogram, sampled_histogram)
from diffpi.kernel import to_positive, from_positive
from diffpi.harness import check_forward_agreement, check_expected_jacobian, run_closure


def unpack(theta):
    return (to_positive(theta[0]), to_positive(theta[1]), theta[2])     # m_R, Gamma, b


def to_params(theta):
    return jnp.array([to_positive(theta[0]), to_positive(theta[1]), theta[2]])


def make_loss(cfg, data, renorm, n_model=None):
    def loss(theta, key):
        p = unpack(theta)
        k1, k2 = jax.random.split(key)
        m1 = weighted_histogram(p, k1, cfg, n_model) * renorm
        m2 = weighted_histogram(p, k2, cfg, n_model) * renorm
        return jnp.mean((m1 - data) * (m2 - data))
    return loss


def main():
    cfg = ConfigBR0()
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_mR, cfg.true_Gamma, cfg.true_b)
    pnames = ["m_R", "Gamma", "b"]

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    # ---- (F) forward agreement ------------------------------------------- #
    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    # ---- (G) expected-histogram Jacobian at true params ------------------ #
    theta0 = jnp.array([from_positive(jnp.asarray(cfg.init_mR)),
                        from_positive(jnp.asarray(cfg.init_Gamma)), cfg.init_b])
    theta_true = jnp.array([from_positive(jnp.asarray(cfg.true_mR)),
                            from_positive(jnp.asarray(cfg.true_Gamma)), cfg.true_b])

    def hist_fn(theta, key):
        return weighted_histogram(unpack(theta), key, cfg, 50_000)

    jc = check_expected_jacobian(hist_fn, theta_true, jax.random.PRNGKey(7),
                                 eps=2e-3, tol=5e-2, n_keys=64)
    print("(G) expected-histogram Jacobian (autodiff vs finite diff, 64 keys):")
    print(f"    occupancy-weighted rel err = {jc.weighted_rel_err:.2e}   "
          f"max rel err = {jc.max_rel_err:.2e}   -> {'PASS' if jc.passed else 'FAIL'}")

    # ---- (C) closure ----------------------------------------------------- #
    print("(C) closure:")
    clo = run_closure(
        make_loss(cfg, data, renorm),
        to_params=to_params, init_unconstrained=theta0,
        key=jax.random.PRNGKey(cfg.fit_seed),
        iterations=cfg.iterations, learning_rate=cfg.learning_rate,
        theta_true=np.array(true_p), clip_norm=5.0e7,
    )
    print("    recovered:")
    for i, nm in enumerate(pnames):
        print(f"      {nm:>6s}: true {true_p[i]:.4f}  ->  fit {clo.theta_fit[i]:.4f}")

    passed = fwd.passed and clo.max_abs_err < 0.02
    print(f"\nB-R0 overall: {'PASS' if passed else 'FAIL'}  (max param err = {clo.max_abs_err:.4f}; "
          f"Jacobian diag weighted-rel-err = {jc.weighted_rel_err:.2f})")

    _plot(cfg, data, clo, renorm, true_p)
    return passed


def _plot(cfg, data, clo, renorm, true_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    init_p = (cfg.init_mR, cfg.init_Gamma, cfg.init_b)
    fit_p = (clo.theta_fit[0], clo.theta_fit[1], clo.theta_fit[2])
    init_h = np.asarray(weighted_histogram(init_p, key, cfg)).reshape(cfg.n_W, cfg.n_z) * renorm
    fit_h = np.asarray(weighted_histogram(fit_p, key, cfg)).reshape(cfg.n_W, cfg.n_z) * renorm
    data_h = np.asarray(data).reshape(cfg.n_W, cfg.n_z)

    Wc = cfg.W_min + (np.arange(cfg.n_W) + 0.5) * (cfg.W_max - cfg.W_min) / cfg.n_W
    zc = -1 + (np.arange(cfg.n_z) + 0.5) * 2 / cfg.n_z

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
    # W projection = sigma(W) line shape
    ax[0].step(Wc, data_h.sum(1), where="mid", color="k", label="data")
    ax[0].step(Wc, init_h.sum(1), where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(Wc, fit_h.sum(1), where="mid", color="tab:blue", label="fit")
    ax[0].axvline(true_p[0], color="gray", ls=":", alpha=0.7)
    ax[0].set(xlabel="W [GeV]", ylabel="events", title="W projection  (σ line shape)")
    ax[0].legend()
    # cos theta projection = angular distribution
    ax[1].step(zc, data_h.sum(0), where="mid", color="k", label="data")
    ax[1].step(zc, init_h.sum(0), where="mid", color="tab:red", ls="--", label="init")
    ax[1].step(zc, fit_h.sum(0), where="mid", color="tab:blue", label="fit")
    ax[1].set(xlabel="cos θ_cm", ylabel="events", title="angular projection")
    ax[1].legend()
    # parameter traces (normalised to truth so they share an axis)
    colors = ["tab:blue", "tab:green", "tab:orange"]
    names = ["m_R", "Gamma", "b"]
    for i in range(3):
        ax[2].plot(clo.param_history[:, i] / (true_p[i] if abs(true_p[i]) > 1e-6 else 1.0),
                   color=colors[i], label=names[i])
    ax[2].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[2].set(xlabel="iteration", ylabel="param / truth", title="parameter traces")
    ax[2].legend(fontsize=8)
    ax[3].plot(clo.loss_history, color="tab:purple")
    ax[3].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[3].set_yscale("log")

    fig.suptitle("Component B — R0 (Δ in elastic π-N, S+P partial waves): recover (m_R, Γ, b)")
    fig.tight_layout(); fig.savefig("b_r0_closure.png", dpi=110)
    print("wrote b_r0_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
