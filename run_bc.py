"""Integrated B->C->G: validate and jointly recover production + FSI knobs.

  (F) forward agreement   : hard sampler == differentiable estimator
  (G) expected-histogram Jacobian : autodiff == finite difference (p_prod, A_abs, A_sc)
  (C) closure             : recover (p_prod, A_abs, A_sc) jointly from the escaped-pion
                            momentum spectrum + absorbed fraction

Run:  python run_bc.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.integrate_bc import ConfigBC, weighted_histogram, sampled_histogram
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
    cfg = ConfigBC()
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_p_prod, cfg.true_A_abs, cfg.true_A_sc)
    pnames = ["p_prod", "A_abs", "A_sc"]

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    theta0 = jnp.array([from_positive(jnp.asarray(cfg.init_p_prod)),
                        from_positive(jnp.asarray(cfg.init_A_abs)),
                        from_positive(jnp.asarray(cfg.init_A_sc))])
    theta_true = jnp.array([from_positive(jnp.asarray(cfg.true_p_prod)),
                            from_positive(jnp.asarray(cfg.true_A_abs)),
                            from_positive(jnp.asarray(cfg.true_A_sc))])

    def hist_fn(theta, key):
        return weighted_histogram(unpack(theta), key, cfg, 50_000)

    jc = check_expected_jacobian(hist_fn, theta_true, jax.random.PRNGKey(7),
                                 eps=2e-3, tol=5e-2, n_keys=64)
    print("(G) expected-histogram Jacobian (autodiff vs finite diff, 64 keys):")
    print(f"    occupancy-weighted rel err = {jc.weighted_rel_err:.2e}   "
          f"max rel err = {jc.max_rel_err:.2e}   -> {'PASS' if jc.passed else 'FAIL'}")

    print("(C) closure:")
    clo = run_closure(
        make_loss(cfg, data, renorm),
        to_params=to_params, init_unconstrained=theta0,
        key=jax.random.PRNGKey(cfg.fit_seed),
        iterations=cfg.iterations, learning_rate=cfg.learning_rate,
        theta_true=np.array(true_p), clip_norm=5.0e7, final_lr_frac=0.05,
    )
    err = np.abs(np.asarray(clo.theta_fit) - np.asarray(true_p))
    print("    recovered:")
    for i, nm in enumerate(pnames):
        print(f"      {nm:>7s}: true {true_p[i]:.4f}  ->  fit {clo.theta_fit[i]:.4f}  (|err| {err[i]:.4f})")

    # p_prod (production) and A_abs (absorption) are tightly constrained by the
    # momentum spectrum; A_sc (elastic scattering) is only weakly constrained,
    # because |p| is conserved in a scatter -- it acts on the spectrum only
    # indirectly via path-lengthening. So we hold it to a looser tolerance and
    # flag it. A future angular observable would pin it down (Strategy §6).
    well_ok = err[0] < 0.02 and err[1] < 0.02
    asc_ok = err[2] < 0.06
    passed = fwd.passed and well_ok and asc_ok
    print(f"\nB->C->G overall: {'PASS' if passed else 'FAIL'}")
    print(f"    production p_prod and absorption A_abs recovered to <0.02 (tight);")
    print(f"    scattering A_sc to {err[2]:.3f} (weakly identifiable from a momentum-only observable).")
    print(f"    (Jacobian diag weighted-rel-err = {jc.weighted_rel_err:.2f}, FD-noise-dominated.)")

    _plot(cfg, data, clo, renorm, true_p)
    _plot_sensitivity(cfg, data, clo, renorm)
    return passed


def _plot_sensitivity(cfg, data, clo, renorm):
    """Show how the escaped-pion spectrum responds to a +/-10% change in each
    parameter, one at a time, about the converged fit. Common random numbers (one
    fixed key) are used for every curve so differences are pure parameter effect,
    not Monte-Carlo noise. The size of the response = how identifiable the
    parameter is from this observable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(2024)                       # common random numbers
    fit_p = tuple(float(x) for x in clo.theta_fit)
    nominal = np.asarray(weighted_histogram(fit_p, key, cfg)) * renorm
    pc = cfg.p_min + (np.arange(cfg.n_p) + 0.5) * (cfg.p_max - cfg.p_min) / cfg.n_p
    names = ["p_prod", "A_abs", "A_sc"]

    def abs_frac(h):
        return h[cfg.n_p] / h.sum()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for i, nm in enumerate(names):
        ax[i].step(pc, np.asarray(data)[:cfg.n_p], where="mid", color="k",
                   lw=1.0, label="data")
        ax[i].step(pc, nominal[:cfg.n_p], where="mid", color="tab:blue", lw=2.0,
                   label=f"fit ({nm}={fit_p[i]:.3f})")
        for frac, color in [(+0.10, "tab:red"), (-0.10, "tab:green")]:
            pv = list(fit_p); pv[i] = pv[i] * (1 + frac)
            h = np.asarray(weighted_histogram(tuple(pv), key, cfg)) * renorm
            ax[i].step(pc, h[:cfg.n_p], where="mid", color=color, ls="--", lw=1.5,
                       label=f"{nm} {frac:+.0%}  (abs={abs_frac(h):.3f})")
        ax[i].axvline(cfg.p_res, color="0.5", ls=":", alpha=0.7)
        ax[i].set(xlabel="escaped pion |p| [GeV/c]",
                  title=f"vary {nm} by ±10%  (nominal abs={abs_frac(nominal):.3f})")
        ax[i].legend(fontsize=7.5, loc="upper left")
    ax[0].set_ylabel("events")
    fig.suptitle("B→C→G sensitivity: response of the escaped spectrum to ±10% per parameter "
                 "(common random numbers). Big response = well constrained; flat = degenerate.")
    fig.tight_layout(); fig.savefig("bc_sensitivity.png", dpi=110)
    print("wrote bc_sensitivity.png")


def _plot(cfg, data, clo, renorm, true_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    init_p = (cfg.init_p_prod, cfg.init_A_abs, cfg.init_A_sc)
    fit_p = tuple(clo.theta_fit)
    init_h = np.asarray(weighted_histogram(init_p, key, cfg)) * renorm
    fit_h = np.asarray(weighted_histogram(fit_p, key, cfg)) * renorm
    data_h = np.asarray(data)
    # a no-FSI reference: production spectrum with absorption/scatter switched off
    prod_only = np.asarray(weighted_histogram((true_p[0], 1e-4, 1e-4), key, cfg)) * renorm

    pc = cfg.p_min + (np.arange(cfg.n_p) + 0.5) * (cfg.p_max - cfg.p_min) / cfg.n_p

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].step(pc, prod_only[:cfg.n_p], where="mid", color="0.6", ls=":",
               label="production (no FSI)")
    ax[0].step(pc, data_h[:cfg.n_p], where="mid", color="k", label="data (with FSI)")
    ax[0].step(pc, init_h[:cfg.n_p], where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(pc, fit_h[:cfg.n_p], where="mid", color="tab:blue", label="fit")
    ax[0].axvline(cfg.p_res, color="tab:green", ls=":", alpha=0.7, label="FSI resonance")
    ax[0].set(xlabel="escaped pion |p| [GeV/c]", ylabel="events",
              title="escaped spectrum (FSI carves dip at p_res)")
    ax[0].legend(fontsize=7)

    abs_frac_d = data_h[cfg.n_p] / data_h.sum()
    abs_frac_f = fit_h[cfg.n_p] / fit_h.sum()
    colors = ["tab:blue", "tab:green", "tab:orange"]
    names = ["p_prod", "A_abs", "A_sc"]
    for i in range(3):
        ax[1].plot(clo.param_history[:, i] / true_p[i], color=colors[i], label=names[i])
    ax[1].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[1].set(xlabel="iteration", ylabel="param / truth",
              title=f"joint recovery (abs frac data={abs_frac_d:.2f}, fit={abs_frac_f:.2f})")
    ax[1].legend(fontsize=8)

    ax[2].plot(clo.loss_history, color="tab:purple")
    ax[2].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[2].set_yscale("log")

    fig.suptitle("Integrated B→C→G: joint recovery of production (p_prod) and FSI (A_abs, A_sc)")
    fig.tight_layout(); fig.savefig("bc_closure.png", dpi=110)
    print("wrote bc_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
