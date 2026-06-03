"""Component D R0: Oset pion absorption, validation + closure.

  (F) forward agreement   : hard sampler == differentiable estimator
  (G) expected-histogram Jacobian : autodiff == finite difference (C_A2, C_A3)
  (C) closure             : recover the 2N and 3N absorption coefficients from the
                            (T_pi, absorption-depth) distribution

Run:  python run_d_r0.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.component_d_r0 import (ConfigDR0, weighted_histogram, sampled_histogram,
                                   density_profile)
from diffpi.kernel import to_positive, from_positive
from diffpi.harness import check_forward_agreement, check_expected_jacobian, run_closure


def unpack(theta):
    return (to_positive(theta[0]), to_positive(theta[1]))


def to_params(theta):
    return jnp.array([to_positive(theta[0]), to_positive(theta[1])])


def make_loss(cfg, data, renorm, n_model=None):
    def loss(theta, key):
        p = unpack(theta)
        k1, k2 = jax.random.split(key)
        m1 = weighted_histogram(p, k1, cfg, n_model) * renorm
        m2 = weighted_histogram(p, k2, cfg, n_model) * renorm
        return jnp.mean((m1 - data) * (m2 - data))
    return loss


def main():
    cfg = ConfigDR0()
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_CA2, cfg.true_CA3)
    pnames = ["C_A2", "C_A3"]

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    theta0 = jnp.array([from_positive(jnp.asarray(cfg.init_CA2)),
                        from_positive(jnp.asarray(cfg.init_CA3))])
    theta_true = jnp.array([from_positive(jnp.asarray(cfg.true_CA2)),
                            from_positive(jnp.asarray(cfg.true_CA3))])

    def hist_fn(theta, key):
        return weighted_histogram(unpack(theta), key, cfg, 50_000)

    jc = check_expected_jacobian(hist_fn, theta_true, jax.random.PRNGKey(7),
                                 eps=2e-3, tol=5e-2, n_keys=48)
    print("(G) expected-histogram Jacobian (autodiff vs finite diff, 48 keys):")
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
        print(f"      {nm:>5s}: true {true_p[i]:.4f}  ->  fit {clo.theta_fit[i]:.4f}  (|err| {err[i]:.4f})")

    passed = fwd.passed and clo.max_abs_err < 0.01
    print(f"\nD-R0 overall: {'PASS' if passed else 'FAIL'}  (max param err = {clo.max_abs_err:.4f}; "
          f"Jacobian diag weighted-rel-err = {jc.weighted_rel_err:.2f})")

    _plot(cfg, data, clo, renorm, true_p)
    return passed


def _plot(cfg, data, clo, renorm, true_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    init_p = (cfg.init_CA2, cfg.init_CA3)
    fit_p = tuple(float(x) for x in clo.theta_fit)
    nb = cfg.K + 1
    init_h = np.asarray(weighted_histogram(init_p, key, cfg)).reshape(cfg.n_T, nb) * renorm
    fit_h = np.asarray(weighted_histogram(fit_p, key, cfg)).reshape(cfg.n_T, nb) * renorm
    data_h = np.asarray(data).reshape(cfg.n_T, nb)

    Tc = cfg.T_min + (np.arange(cfg.n_T) + 0.5) * (cfg.T_max - cfg.T_min) / cfg.n_T

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
    # transmitted fraction vs T (the "transparency")
    ax[0].step(Tc, data_h[:, cfg.K] / data_h.sum(1), where="mid", color="k", label="data")
    ax[0].step(Tc, init_h[:, cfg.K] / init_h.sum(1), where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(Tc, fit_h[:, cfg.K] / fit_h.sum(1), where="mid", color="tab:blue", label="fit")
    ax[0].set(xlabel="T_pi [GeV]", ylabel="transmitted fraction",
              title="transparency vs energy"); ax[0].legend()
    # absorption-depth distribution (summed over T)
    depth = np.arange(cfg.K)
    ax[1].step(depth, data_h[:, :cfg.K].sum(0), where="mid", color="k", label="data")
    ax[1].step(depth, init_h[:, :cfg.K].sum(0), where="mid", color="tab:red", ls="--", label="init")
    ax[1].step(depth, fit_h[:, :cfg.K].sum(0), where="mid", color="tab:blue", label="fit")
    ax[1].set(xlabel="absorption depth (step)", ylabel="events",
              title="absorption depth (rho-power)"); ax[1].legend()
    # parameter traces
    colors = ["tab:blue", "tab:orange"]
    names = ["C_A2", "C_A3"]
    for i in range(2):
        ax[2].plot(clo.param_history[:, i] / true_p[i], color=colors[i], label=names[i])
    ax[2].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[2].set(xlabel="iteration", ylabel="param / truth", title="parameter traces"); ax[2].legend(fontsize=8)
    ax[3].plot(clo.loss_history, color="tab:purple")
    ax[3].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[3].set_yscale("log")

    fig.suptitle("Component D — R0 (Oset absorption): recover 2N/3N coefficients C_A2, C_A3")
    fig.tight_layout(); fig.savefig("d_r0_closure.png", dpi=110)
    print("wrote d_r0_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
