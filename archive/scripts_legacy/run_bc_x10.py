"""Integrated B->C->G with ~10x statistics, to test whether the A_sc residual is
noise-limited (should shrink) or identifiability-limited (should not).

Same physics/estimator as run_bc.py (unchanged); only n_data, n_model are scaled.

Run:  python run_bc_x10.py
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import jax

from diffpi.integrate_bc import ConfigBC, weighted_histogram, sampled_histogram
from diffpi.kernel import from_positive
from diffpi.harness import check_forward_agreement, run_closure
from run_bc import unpack, to_params, make_loss, _plot_sensitivity


def main():
    cfg = replace(ConfigBC(), n_data=4_000_000, n_model=50_000,
                  iterations=300, learning_rate=0.03)
    renorm = cfg.n_data / cfg.n_model
    true_p = (cfg.true_p_prod, cfg.true_A_abs, cfg.true_A_sc)
    pnames = ["p_prod", "A_abs", "A_sc"]
    print(f"n_data={cfg.n_data:,}  n_model={cfg.n_model:,}  iters={cfg.iterations}")

    data = jax.lax.stop_gradient(
        sampled_histogram(true_p, jax.random.PRNGKey(cfg.data_seed), cfg))

    model = weighted_histogram(true_p, jax.random.PRNGKey(0), cfg) * renorm
    fwd = check_forward_agreement(data, model, cfg.n_data, cfg.n_model)
    print(f"(F) forward agreement: max|hard-model|={fwd.max_abs_diff:.1f} "
          f"tol={fwd.mc_tol:.1f} -> {'PASS' if fwd.passed else 'FAIL'}")

    theta0 = np.array([from_positive(np.asarray(cfg.init_p_prod)),
                       from_positive(np.asarray(cfg.init_A_abs)),
                       from_positive(np.asarray(cfg.init_A_sc))])

    print("(C) closure (x10 stats):")
    clo = run_closure(
        make_loss(cfg, data, renorm), to_params=to_params, init_unconstrained=theta0,
        key=jax.random.PRNGKey(cfg.fit_seed),
        iterations=cfg.iterations, learning_rate=cfg.learning_rate,
        theta_true=np.array(true_p), clip_norm=5.0e8, final_lr_frac=0.05,
    )
    err = np.abs(np.asarray(clo.theta_fit) - np.asarray(true_p))
    for i, nm in enumerate(pnames):
        print(f"    {nm:>7s}: true {true_p[i]:.4f} -> fit {clo.theta_fit[i]:.4f}  (|err| {err[i]:.4f})")
    print(f"\n    A_sc |err|: x1 run was 0.0130; x10 run is {err[2]:.4f}")

    _plot(cfg, clo, true_p)
    _plot_sensitivity(cfg, data, clo, renorm)     # ±10%-per-parameter response (bc_sensitivity.png)
    return err


def _plot(cfg, clo, true_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = ["tab:blue", "tab:green", "tab:orange"]
    names = ["p_prod", "A_abs", "A_sc"]
    for i in range(3):
        ax[0].plot(clo.param_history[:, i] / true_p[i], color=colors[i], label=names[i])
    ax[0].axhline(1.0, color="gray", ls=":", alpha=0.7)
    ax[0].set(xlabel="iteration", ylabel="param / truth",
              title=f"x10 stats (n_data={cfg.n_data:,})")
    ax[0].legend(fontsize=8)
    ax[1].plot(clo.loss_history, color="tab:purple")
    ax[1].set(xlabel="iteration", ylabel="loss", title="loss"); ax[1].set_yscale("log")
    fig.suptitle("B→C→G with ~10x statistics: is A_sc noise- or identifiability-limited?")
    fig.tight_layout(); fig.savefig("bc_x10_closure.png", dpi=110)
    print("wrote bc_x10_closure.png")


if __name__ == "__main__":
    main()
