"""Component E R0: propagating Delta, validation + closure.

  (F) forward agreement   : hard sampler == differentiable estimator
  (G) expected-histogram Jacobian : autodiff == finite difference (m_Delta, Gamma, sigma_abs)
  (C) closure             : recover the Delta mass, width, and absorption strength from the
                            (off-shell mass, decay-time / absorbed) distribution

Run:  python run_e_r0.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from diffpi.component_e_r0 import ConfigER0, weighted_histogram, sampled_histogram
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
    cfg = ConfigER0()
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_mDelta, cfg.true_Gamma, cfg.true_sigma_abs)
    pnames = ["m_Delta", "Gamma", "sigma_abs"]

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print("(F) forward agreement:")
    print(f"    max|hard-model| = {fwd.max_abs_diff:.2f}   MC tol (5σ) = {fwd.mc_tol:.2f}   "
          f"-> {'PASS' if fwd.passed else 'FAIL'}")

    theta0 = jnp.array([from_positive(jnp.asarray(cfg.init_mDelta)),
                        from_positive(jnp.asarray(cfg.init_Gamma)),
                        from_positive(jnp.asarray(cfg.init_sigma_abs))])
    theta_true = jnp.array([from_positive(jnp.asarray(cfg.true_mDelta)),
                            from_positive(jnp.asarray(cfg.true_Gamma)),
                            from_positive(jnp.asarray(cfg.true_sigma_abs))])

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
        print(f"      {nm:>9s}: true {true_p[i]:.4f}  ->  fit {clo.theta_fit[i]:.4f}  (|err| {err[i]:.4f})")

    passed = fwd.passed and clo.max_abs_err < 0.02
    print(f"\nE-R0 overall: {'PASS' if passed else 'FAIL'}  (max param err = {clo.max_abs_err:.4f}; "
          f"Jacobian diag weighted-rel-err = {jc.weighted_rel_err:.2f})")

    _plot(cfg, data, clo, renorm, true_p)
    return passed


def _plot(cfg, data, clo, renorm, true_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(321)
    init_p = (cfg.init_mDelta, cfg.init_Gamma, cfg.init_sigma_abs)
    fit_p = tuple(float(x) for x in clo.theta_fit)
    no = cfg.n_out
    init_h = np.asarray(weighted_histogram(init_p, key, cfg)).reshape(cfg.n_mu, no) * renorm
    fit_h = np.asarray(weighted_histogram(fit_p, key, cfg)).reshape(cfg.n_mu, no) * renorm
    data_h = np.asarray(data).reshape(cfg.n_mu, no)

    muc = cfg.mu_min + (np.arange(cfg.n_mu) + 0.5) * (cfg.mu_max - cfg.mu_min) / cfg.n_mu
    tc = (np.arange(cfg.K) + 0.5) * cfg.dt

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))
    # off-shell mass distribution (spectral function): decays summed over time
    ax[0].step(muc, data_h[:, :cfg.K].sum(1), where="mid", color="k", label="data")
    ax[0].step(muc, init_h[:, :cfg.K].sum(1), where="mid", color="tab:red", ls="--", label="init")
    ax[0].step(muc, fit_h[:, :cfg.K].sum(1), where="mid", color="tab:blue", label="fit")
    ax[0].axvline(true_p[0], color="0.5", ls=":", alpha=0.7)
    ax[0].set(xlabel="off-shell mass μ [GeV]", ylabel="decays",
              title="spectral function (sets m_Δ, Γ)"); ax[0].legend()
    # decay-time distribution (summed over mu)
    ax[1].step(tc, data_h[:, :cfg.K].sum(0), where="mid", color="k", label="data")
    ax[1].step(tc, init_h[:, :cfg.K].sum(0), where="mid", color="tab:red", ls="--", label="init")
    ax[1].step(tc, fit_h[:, :cfg.K].sum(0), where="mid", color="tab:blue", label="fit")
    ax[1].set(xlabel="decay time [fm/c]", ylabel="decays", title="decay time (sets Γ)")
    ax[1].set_yscale("log"); ax[1].legend()
    # parameter traces + absorbed fraction
    abs_d = data_h[:, cfg.K].sum() / data_h.sum()
    abs_f = fit_h[:, cfg.K].sum() / fit_h.sum()
    colors = ["tab:blue", "tab:green", "tab:orange"]
    names = ["m_Δ", "Γ", "σ_abs"]
    for i in range(3):
        ax[2].plot(clo.param_history[:, i] / true_p[i], color=colors[i], label=names[i])
    ax[2].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[2].set(xlabel="iteration", ylabel="param / truth",
              title=f"traces (absorbed frac data={abs_d:.2f}, fit={abs_f:.2f})")
    ax[2].legend(fontsize=8)
    ax[3].plot(clo.loss_history, color="tab:purple")
    ax[3].set(xlabel="iteration", ylabel="MSE loss", title="loss"); ax[3].set_yscale("log")

    fig.suptitle("Component E — R0 (propagating Δ): recover m_Δ, Γ, σ_abs from decay-vs-absorption")
    fig.tight_layout(); fig.savefig("e_r0_closure.png", dpi=110)
    print("wrote e_r0_closure.png")


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
