"""Section 4 convergence figure: the LM/Gauss-Newton optimization trajectory of the multisample closure.

Reads the trajectory persisted by analysis.paper.physfit.multisample (traj_theta/traj_chi2/... in the run
npz) -- no re-fit.  Two panels:

  (a) objective vs step        -- chi2_total (+prior) and chi2_data on a log axis.
  (b) distance to truth vs step-- per dial |theta_k - theta*_k| / sigma_prior_k on a log axis, coloured by
      physics block.  Every line starts at its injected displacement and collapses toward zero, so the 16
      curves read as one converging bundle (clearer than 16 signed trajectories crossing each other).

Usage:  python -m analysis.paper.sec4_closure.convergence [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style


def main(label="sec4_closure_random16_s10"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    if "traj_theta" not in z.files:
        raise SystemExit(f"{label}.npz has no trajectory (re-run multisample with the record hook).")
    pnames = [str(x) for x in z["pnames"]]
    subset = [int(k) for k in z["subset"]]
    truth = np.asarray(z["truth"]); prior = np.asarray(z["prior"])
    th = np.asarray(z["traj_theta"]); c2 = np.asarray(z["traj_chi2"]); c2d = np.asarray(z["traj_chi2data"])
    nstep = len(c2) - 1                                          # points include the start (step 0)
    step = np.arange(len(c2))

    order = sorted(subset, key=lambda k: (style.knob_group(pnames[k]), k))
    dist = {k: np.abs(th[:, k] - truth[k]) / max(prior[k], 1e-12) for k in order}   # |dtheta|/sigma_prior

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))

        a.semilogy(step, c2, "-o", ms=3.5, lw=1.3, color="#1f4b9c", label=r"$\chi^2$ total (+prior)")
        a.semilogy(step, c2d, "-s", ms=3.0, lw=1.3, color=style.FIT_EC, label=r"$\chi^2_{\rm data}$")
        a.set_xlabel("LM / Gauss-Newton step"); a.set_ylabel(r"$\chi^2$")
        a.legend(fontsize=9); a.set_title("(a)  objective vs step", loc="left", fontsize=10.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

        seen = set()
        for k in order:
            g = style.knob_group(pnames[k]); c = style.KNOB_GROUP_COLOR[g]
            lab = style.KNOB_GROUP_NAME[g].replace("\n", " ") if g not in seen else None
            seen.add(g)
            b.semilogy(step, np.maximum(dist[k], 1e-4), "-", lw=1.2, color=c, alpha=0.9, label=lab)
        b.set_xlabel("LM / Gauss-Newton step")
        b.set_ylabel(r"distance to truth  $|\theta-\theta^\star|/\sigma_{\rm prior}$")
        b.set_title(f"(b)  16 dials converging to truth  ({nstep} steps)", loc="left", fontsize=10.5)
        b.legend(fontsize=8, title="physics block", title_fontsize=8, loc="upper right", framealpha=0.9)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)

        fig.suptitle(f"S4 closure convergence — {label}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        style.save(fig, f"{label}_convergence")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
