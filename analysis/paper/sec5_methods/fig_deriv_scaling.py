"""Sec 5 -- derivative-order scaling: measured autodiff cost vs the combinatorial x exponential theory.

Reads deriv_scaling.npz (from deriv_scaling.py).  Two panels:
  (left)  per-term cost t_term(k) of one depth-k nested jvp -- measured points + the fitted rho^(k-1) law
          (each forward-over-forward layer re-differentiates the graph, so cost multiplies by ~rho/order);
  (right) full order-k tensor build time T_k = N_k * t_term(k): measured where cheap (k<=BUILD_MAX) vs the
          prediction N_k * t_term(k), with N_k = C(nsub+k-1,k).  The product is combinatorial x exponential,
          so the wall-clock crosses seconds -> minutes -> hours -> days within a few orders.

Usage:  python -m analysis.paper.sec5_methods.fig_deriv_scaling
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style


def _hms(s):
    if s < 90: return f"{s:.0f}s"
    if s < 5400: return f"{s/60:.0f}min"
    if s < 1.5 * 86400: return f"{s/3600:.1f}hr"
    return f"{s/86400:.1f}d"


def main():
    style.use()
    z = np.load(style.ALTGEN / "deriv_scaling.npz", allow_pickle=True)
    k = np.asarray(z["order"]); N = np.asarray(z["n_terms"]); tt = np.asarray(z["t_term"])
    tstd = np.asarray(z["t_term_std"]); pred = np.asarray(z["pred_full"]); meas = np.asarray(z["t_full"])
    nsub = int(z["nsub"])

    # per-term exponential fit  t_term ~ t1 * rho^(k-1)  (log-linear LS on all measured orders)
    b, a = np.polyfit(k, np.log(tt), 1); rho = float(np.exp(b)); t1 = float(np.exp(a + b))
    fit = np.exp(a + b * k)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- left: per-term cost ----------------------------------------------------------------------- #
    axL.errorbar(k, tt * 1e3, yerr=tstd * 1e3, fmt="o", color="#1f4b9c", ms=6, capsize=3, label="measured")
    axL.plot(k, fit * 1e3, "--", color="#2e7d32", lw=1.4,
             label=fr"fit $t_1\,\rho^{{k-1}}$, $\rho={rho:.2f}$")
    axL.set_yscale("log"); axL.set_xlabel("derivative order $k$"); axL.set_ylabel("cost of one term  [ms]")
    axL.set_xticks(k); axL.set_title("per-term: one depth-$k$ nested jvp", fontsize=10, loc="left")
    axL.legend(fontsize=8.5); axL.grid(alpha=0.25, which="both")

    # ---- right: full-build time -------------------------------------------------------------------- #
    axR.plot(k, N, ":", color="0.6", lw=1.2)
    axR2 = axR.twinx(); axR2.set_yscale("log"); axR2.set_ylabel(r"entries $N_k=\binom{n+k-1}{k}$", color="0.5")
    axR2.plot(k, N, ":", color="0.6", lw=1.2, label="_"); axR2.tick_params(colors="0.5")
    axR.set_yscale("log")
    axR.plot(k, pred, "-", color="#1f4b9c", lw=1.5, marker="s", ms=5, label=r"predicted $N_k\,t_{\rm term}(k)$")
    mmask = np.isfinite(meas)
    axR.plot(k[mmask], meas[mmask], "*", color="#c1121f", ms=15, label="measured full build", zorder=5)
    for kk, pp in zip(k, pred):
        axR.annotate(_hms(pp), (kk, pp), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7.5, color="#1f4b9c")
    axR.set_xlabel("derivative order $k$"); axR.set_ylabel("full order-$k$ build  [s]")
    axR.set_xticks(k); axR.set_title(fr"full tensor: $T_k=N_k\,t_{{\rm term}}(k)$  ($n={nsub}$ dials)",
                                     fontsize=10, loc="left")
    axR.legend(fontsize=8.5, loc="upper left"); axR.grid(alpha=0.25, which="both")

    fig.suptitle("Sec 5  autodiff derivative cost vs order — measured vs combinatorial$\\times$exponential theory",
                 fontsize=11.5)
    fig.tight_layout()
    style.save(fig, "sec5_deriv_scaling")


if __name__ == "__main__":
    main()
