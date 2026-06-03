"""Component C R0: run the three validation gates and a closure plot.

  (F) forward agreement : hard sampler  ==  differentiable estimator
  (G) gradient check    : autodiff      ==  central finite differences
  (C) closure           : recover the true mean free path from a histogram

Run:  python run_c_r0.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.component_c_r0 import (ConfigR0, weighted_histogram, sampled_histogram)
from diffpi.kernel import to_positive, from_positive
from diffpi.harness import check_forward_agreement, check_gradient, run_closure


def main():
    cfg = ConfigR0()
    renorm = cfg.n_data / cfg.n_model

    # ---- data: hard sampler at the true L -------------------------------- #
    data = jax.lax.stop_gradient(
        sampled_histogram(cfg.L_true, jax.random.PRNGKey(cfg.data_seed), cfg))

    # ---- (F) forward agreement ------------------------------------------- #
    model = weighted_histogram(cfg.L_true, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.3f}   "
          f"MC tol (5σ) = {fwd.mc_tol:.3f}   -> {'PASS' if fwd.passed else 'FAIL'}")

    # ---- (G) gradient check ---------------------------------------------- #
    def loss(theta, key):                      # theta unconstrained, scalar in array
        L = to_positive(theta[0])
        m = weighted_histogram(L, key, cfg) * renorm
        return jnp.mean((m - data) ** 2)

    theta0 = jnp.asarray([from_positive(jnp.asarray(cfg.L_init))])
    gc = check_gradient(loss, theta0, jax.random.PRNGKey(7))
    print("(G) gradient check:")
    print(f"    autodiff={gc.autodiff[0]:+.6e}  finite_diff={gc.finite_diff[0]:+.6e}  "
          f"rel_err={gc.max_rel_err:.2e}  -> {'PASS' if gc.passed else 'FAIL'}")

    # ---- (C) closure ----------------------------------------------------- #
    print("(C) closure:")
    clo = run_closure(
        loss,
        to_params=lambda t: to_positive(t),
        init_unconstrained=theta0,
        key=jax.random.PRNGKey(cfg.fit_seed),
        iterations=cfg.iterations, learning_rate=cfg.learning_rate,
        theta_true=np.array([cfg.L_true]),
    )
    print(f"    true L = {cfg.L_true:.4f}   ->   fit L = {clo.theta_fit[0]:.4f}   "
          f"|err| = {clo.max_abs_err:.4f}")

    overall = fwd.passed and gc.passed and clo.max_abs_err < 0.05
    print(f"\nR0 overall: {'PASS' if overall else 'FAIL'}")

    _plot(cfg, data, clo, renorm)
    return overall


def _plot(cfg, data, clo, renorm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(123)
    init_hist = np.asarray(weighted_histogram(jnp.asarray(cfg.L_init), key, cfg)) * renorm
    final_hist = np.asarray(weighted_histogram(jnp.asarray(clo.theta_fit[0]), key, cfg)) * renorm
    bins = np.arange(cfg.n_bins)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].step(bins, np.asarray(data), where="mid", label="data (hard)", color="k")
    ax[0].step(bins, init_hist, where="mid", label=f"init L={cfg.L_init}", color="tab:red", ls="--")
    ax[0].step(bins, final_hist, where="mid", label=f"fit L={clo.theta_fit[0]:.3f}", color="tab:blue")
    ax[0].set(xlabel="interaction-depth bin (K=transmitted)", ylabel="counts",
              title="R0 closure: histogram"); ax[0].legend(); ax[0].set_yscale("log")

    ax[1].plot(clo.param_history[:, 0], color="tab:blue")
    ax[1].axhline(cfg.L_true, color="k", ls=":", label="true")
    ax[1].set(xlabel="iteration", ylabel="L [fm]", title="parameter trace"); ax[1].legend()

    ax[2].plot(clo.loss_history, color="tab:purple")
    ax[2].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[2].set_yscale("log")

    fig.suptitle("Component C — R0 (1-D slab, single channel): recover mean free path L")
    fig.tight_layout()
    fig.savefig("c_r0_closure.png", dpi=110)
    print("wrote c_r0_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
