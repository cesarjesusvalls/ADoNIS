"""Component A R0: hard vertex (CC single-pion production), validation + closure.

  (F) forward agreement   : hard sampler == differentiable (reweighting) estimator
  (G) expected-histogram Jacobian : autodiff == finite difference (M_A, m_Delta, Gamma_Delta)
  (C) closure             : recover the axial mass M_A and the Delta mass/width from the
                            2-D (Q^2, W) distribution

Run:  python run_a_r0.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.component_a_r0 import (ConfigAR0, weighted_histogram, sampled_histogram, rate)
from diffpi.kernel import to_positive, from_positive
from diffpi.harness import check_forward_agreement, check_expected_jacobian, run_closure


def unpack(theta):
    return (to_positive(theta[0]), to_positive(theta[1]), to_positive(theta[2]))


def to_params(theta):
    return jnp.array([to_positive(theta[0]), to_positive(theta[1]), to_positive(theta[2])])


def make_loss(cfg, data, renorm, n_model=None):
    def loss(theta, key):
        p = unpack(theta)
        k1, k2 = jax.random.split(key)
        m1 = weighted_histogram(p, k1, cfg, n_model) * renorm
        m2 = weighted_histogram(p, k2, cfg, n_model) * renorm
        return jnp.mean((m1 - data) * (m2 - data))
    return loss


def main():
    cfg = ConfigAR0()
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_MA, cfg.true_mDelta, cfg.true_GammaDelta)
    pnames = ["M_A", "m_Delta", "Gamma_Delta"]

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    theta0 = jnp.array([from_positive(jnp.asarray(cfg.init_MA)),
                        from_positive(jnp.asarray(cfg.init_mDelta)),
                        from_positive(jnp.asarray(cfg.init_GammaDelta))])
    theta_true = jnp.array([from_positive(jnp.asarray(cfg.true_MA)),
                            from_positive(jnp.asarray(cfg.true_mDelta)),
                            from_positive(jnp.asarray(cfg.true_GammaDelta))])

    def hist_fn(theta, key):
        return weighted_histogram(unpack(theta), key, cfg, 50_000)

    jc = check_expected_jacobian(hist_fn, theta_true, jax.random.PRNGKey(7),
                                 eps=2e-3, tol=5e-2, n_keys=64)
    print("(G) expected-histogram Jacobian (autodiff vs finite diff, 64 keys):")
    print(f"    occupancy-weighted rel err = {jc.weighted_rel_err:.2e}   "
          f"max rel err = {jc.max_rel_err:.2e}   -> {'PASS' if jc.passed else 'FAIL'}")

    print("(C) closure:")
    clo = run_closure(
        make_loss(cfg, data, renorm), to_params=to_params, init_unconstrained=theta0,
        key=jax.random.PRNGKey(cfg.fit_seed),
        iterations=cfg.iterations, learning_rate=cfg.learning_rate,
        theta_true=np.array(true_p), clip_norm=5.0e7, final_lr_frac=0.1,
    )
    err = np.abs(np.asarray(clo.theta_fit) - np.asarray(true_p))
    print("    recovered:")
    for i, nm in enumerate(pnames):
        print(f"      {nm:>12s}: true {true_p[i]:.4f}  ->  fit {clo.theta_fit[i]:.4f}  (|err| {err[i]:.4f})")

    passed = fwd.passed and clo.max_abs_err < 0.02
    print(f"\nA-R0 overall: {'PASS' if passed else 'FAIL'}  (max param err = {clo.max_abs_err:.4f}; "
          f"Jacobian diag weighted-rel-err = {jc.weighted_rel_err:.2f})")

    _plot(cfg, data, clo, renorm, true_p)
    return passed


def _plot(cfg, data, clo, renorm, true_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    init_p = (cfg.init_MA, cfg.init_mDelta, cfg.init_GammaDelta)
    fit_p = tuple(float(x) for x in clo.theta_fit)
    init_h = np.asarray(weighted_histogram(init_p, key, cfg)).reshape(cfg.n_Q2, cfg.n_W) * renorm
    fit_h = np.asarray(weighted_histogram(fit_p, key, cfg)).reshape(cfg.n_Q2, cfg.n_W) * renorm
    data_h = np.asarray(data).reshape(cfg.n_Q2, cfg.n_W)

    Q2c = (np.arange(cfg.n_Q2) + 0.5) * cfg.Q2_max / cfg.n_Q2
    Wc = cfg.W_min + (np.arange(cfg.n_W) + 0.5) * (cfg.W_max - cfg.W_min) / cfg.n_W

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
    # Q^2 projection -> axial mass (steepness)
    ax[0].step(Q2c, data_h.sum(1), where="mid", color="k", label="data")
    ax[0].step(Q2c, init_h.sum(1), where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(Q2c, fit_h.sum(1), where="mid", color="tab:blue", label="fit")
    ax[0].set(xlabel="Q^2 [GeV^2]", ylabel="events", title="Q^2 projection (sets M_A)")
    ax[0].set_yscale("log"); ax[0].legend()
    # W projection -> Delta mass/width
    ax[1].step(Wc, data_h.sum(0), where="mid", color="k", label="data")
    ax[1].step(Wc, init_h.sum(0), where="mid", color="tab:red", ls="--", label="init")
    ax[1].step(Wc, fit_h.sum(0), where="mid", color="tab:blue", label="fit")
    ax[1].axvline(true_p[1], color="0.5", ls=":", alpha=0.7)
    ax[1].set(xlabel="W [GeV]", ylabel="events", title="W projection (Delta peak)"); ax[1].legend()
    # parameter traces (param / truth)
    colors = ["tab:blue", "tab:green", "tab:orange"]
    names = ["M_A", "m_Delta", "Gamma_Delta"]
    for i in range(3):
        ax[2].plot(clo.param_history[:, i] / true_p[i], color=colors[i], label=names[i])
    ax[2].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[2].set(xlabel="iteration", ylabel="param / truth", title="parameter traces"); ax[2].legend(fontsize=8)
    ax[3].plot(clo.loss_history, color="tab:purple")
    ax[3].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[3].set_yscale("log")

    fig.suptitle("Component A — R0 (CC single-pion production): recover axial mass M_A and Delta m/Γ")
    fig.tight_layout(); fig.savefig("a_r0_closure.png", dpi=110)
    print("wrote a_r0_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
