"""Full integrated chain A->(C+B+D)->G: joint recovery of vertex + FSI parameters.

  (F) forward agreement   : hard sampler == differentiable estimator
  (G) expected-histogram Jacobian : autodiff == finite difference (5 params)
  (C) closure             : recover M_A, m_Delta, Gamma_Delta, sigma_scatter, sigma_abs
                            jointly from the (Q^2, final pion |p|) + CC0pi observable

Run:  python run_full.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.integrate_full import ConfigFull, weighted_histogram, sampled_histogram
from diffpi.kernel import to_positive, from_positive
from diffpi.harness import check_forward_agreement, check_expected_jacobian, run_closure

PNAMES = ["M_A", "m_Delta", "Gamma_Delta", "sigma_scatter", "sigma_abs"]


def unpack(theta):
    return tuple(to_positive(theta[i]) for i in range(5))


def to_params(theta):
    return jnp.array([to_positive(theta[i]) for i in range(5)])


def make_loss(cfg, data, renorm, n_model=None):
    def loss(theta, key):
        p = unpack(theta)
        k1, k2 = jax.random.split(key)
        m1 = weighted_histogram(p, k1, cfg, n_model) * renorm
        m2 = weighted_histogram(p, k2, cfg, n_model) * renorm
        return jnp.mean((m1 - data) * (m2 - data))
    return loss


def main():
    cfg = ConfigFull()
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_MA, cfg.true_mDelta, cfg.true_GammaDelta,
              cfg.true_sigma_scatter, cfg.true_sigma_abs)
    init_p = (cfg.init_MA, cfg.init_mDelta, cfg.init_GammaDelta,
              cfg.init_sigma_scatter, cfg.init_sigma_abs)

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    theta0 = jnp.array([from_positive(jnp.asarray(v)) for v in init_p])
    theta_true = jnp.array([from_positive(jnp.asarray(v)) for v in true_p])

    def hist_fn(theta, key):
        return weighted_histogram(unpack(theta), key, cfg, 60_000)

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
        theta_true=np.array(true_p), clip_norm=5.0e7, final_lr_frac=0.25,
    )
    err = np.abs(np.asarray(clo.theta_fit) - np.asarray(true_p))
    print("    recovered:")
    for i, nm in enumerate(PNAMES):
        print(f"      {nm:>14s}: true {true_p[i]:.4f}  ->  fit {clo.theta_fit[i]:.4f}  (|err| {err[i]:.4f})")

    passed = fwd.passed and np.max(err) < 0.05
    print(f"\nFULL A->(C+B+D)->G overall: {'PASS' if passed else 'FAIL'}  "
          f"(max param err = {np.max(err):.4f}; Jacobian diag = {jc.weighted_rel_err:.2f})")

    _plot(cfg, data, clo, renorm, true_p, init_p)
    _plot_evolution(clo, true_p, init_p)
    return passed


def _plot_evolution(clo, true_p, init_p):
    """One panel per parameter: its absolute value vs iteration, with truth line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.2))
    for i, nm in enumerate(PNAMES):
        ax[i].plot(clo.param_history[:, i], color="tab:blue", lw=1.8)
        ax[i].axhline(true_p[i], color="k", ls="--", lw=1.2, label=f"true {true_p[i]:.3f}")
        ax[i].axhline(init_p[i], color="tab:red", ls=":", lw=1.0, alpha=0.6, label=f"init {init_p[i]:.3f}")
        ax[i].set(xlabel="iteration", title=f"{nm}\nfit {clo.theta_fit[i]:.3f}")
        ax[i].legend(fontsize=7, loc="best")
    fig.suptitle("Full chain — parameter evolution during the joint fit (blue), with truth (dashed) and init (dotted)")
    fig.tight_layout(); fig.savefig("full_evolution.png", dpi=110)
    print("wrote full_evolution.png")


def _plot(cfg, data, clo, renorm, true_p, init_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    fit_p = tuple(float(x) for x in clo.theta_fit)
    no = cfg.n_out
    dh = np.asarray(data).reshape(cfg.n_Q2, no)
    ih = np.asarray(weighted_histogram(init_p, key, cfg)).reshape(cfg.n_Q2, no) * renorm
    fh = np.asarray(weighted_histogram(fit_p, key, cfg)).reshape(cfg.n_Q2, no) * renorm

    Q2c = (np.arange(cfg.n_Q2) + 0.5) * cfg.Q2_max / cfg.n_Q2
    ppc = (np.arange(cfg.n_ppi) + 0.5) * cfg.p_pi_max / cfg.n_ppi

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    # Q^2 projection (escaped + absorbed) -> M_A
    ax[0].step(Q2c, dh.sum(1), where="mid", color="k", label="data")
    ax[0].step(Q2c, ih.sum(1), where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(Q2c, fh.sum(1), where="mid", color="tab:blue", label="fit")
    ax[0].set(xlabel="Q^2 [GeV^2]", ylabel="events", title="Q^2 (sets M_A)")
    ax[0].set_yscale("log"); ax[0].legend()
    # final pion momentum (escaped) -> resonance + FSI
    ax[1].step(ppc, dh[:, :cfg.n_ppi].sum(0), where="mid", color="k", label="data")
    ax[1].step(ppc, ih[:, :cfg.n_ppi].sum(0), where="mid", color="tab:red", ls="--", label="init")
    ax[1].step(ppc, fh[:, :cfg.n_ppi].sum(0), where="mid", color="tab:blue", label="fit")
    ax[1].set(xlabel="final pion |p| [GeV/c]", ylabel="CC1π events",
              title="pion momentum (resonance + FSI)"); ax[1].legend()
    # CC0pi (absorbed) fraction vs Q^2
    ax[2].step(Q2c, dh[:, cfg.n_ppi] / dh.sum(1), where="mid", color="k", label="data")
    ax[2].step(Q2c, fh[:, cfg.n_ppi] / fh.sum(1), where="mid", color="tab:blue", label="fit")
    ax[2].set(xlabel="Q^2 [GeV^2]", ylabel="CC0π (absorbed) fraction",
              title="absorbed fraction (sets σ_abs)"); ax[2].legend()
    # parameter traces
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    for i in range(5):
        ax[3].plot(clo.param_history[:, i] / true_p[i], color=colors[i], label=PNAMES[i])
    ax[3].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[3].set(xlabel="iteration", ylabel="param / truth", title="joint recovery (5 params)")
    ax[3].legend(fontsize=7)

    fig.suptitle("FULL chain A→(C+B+D)→G: joint recovery of vertex (M_A, m_Δ, Γ) and FSI (σ_sc, σ_abs)")
    fig.tight_layout(); fig.savefig("full_closure.png", dpi=110)
    print("wrote full_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
