"""Degeneracy-breaking overlay: qe_norm x res_norm 1sigma/2sigma constraint from CC0pi-only vs CC0pi+CC1pi.

Both norm surfaces are EXACTLY quadratic (a per-channel normalization is a linear scale on the model), so the
Hessian ellipse IS the exact Delta chi^2 contour -- the comparison is drawn straight from the saved (BFP, V=2H^-1)
of each run, no recompute.  Shows that CC1pi+Np (which directly measures RES) collapses the res_norm direction
that CC0pi alone leaves loose.

  python scripts/cc0pi_cc1pi_compare.py
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUTDIR = "/tmp/adonis_llh"


def _ellipse(ax, bfp, V, color, label, lw=2.0):
    th = np.linspace(0, 2 * np.pi, 200)
    w, Vv = np.linalg.eigh(V)
    for k, ls in ((1.0, "-"), (2.0, "--")):
        xy = Vv @ (np.sqrt(np.maximum(w, 0) * k)[:, None] * np.array([np.cos(th), np.sin(th)]))
        ax.plot(bfp[0] + xy[0], bfp[1] + xy[1], color=color, ls=ls, lw=lw,
                label=f"{label} {int(k)}$\\sigma$" if k == 1 else None)


def main():
    d0 = np.load(f"{OUTDIR}/qe_res_norm_dptdat.npz", allow_pickle=True)     # CC0pi only
    dc = np.load(f"{OUTDIR}/qe_res_norm_cc0cc1.npz", allow_pickle=True)     # CC0pi + CC1pi
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    _ellipse(ax, d0["bfp"], d0["V"], "C0", "CC0$\\pi$ (dpt+dat)")
    _ellipse(ax, dc["bfp"], dc["V"], "C3", "CC0$\\pi$+CC1$\\pi$")
    ax.plot(*d0["bfp"], "o", color="C0", ms=7, mec="k")
    ax.plot(*dc["bfp"], "X", color="C3", ms=11, mec="k")
    ax.plot(1, 1, "s", color="0.5", ms=8, mec="k", label="nominal")
    s0 = np.sqrt(np.diag(d0["V"])); sc = np.sqrt(np.diag(dc["V"]))
    ax.set_xlabel("QE norm"); ax.set_ylabel("RES norm")
    ax.set_title("Adding CC1$\\pi^+$Np breaks the CC0$\\pi$ RES-norm degeneracy\n"
                 f"$\\sigma_{{\\rm RES}}$: {s0[1]:.2f} (CC0$\\pi$) $\\to$ {sc[1]:.2f} (CC0$\\pi$+CC1$\\pi$);   "
                 f"$\\sigma_{{\\rm QE}}$: {s0[0]:.3f} $\\to$ {sc[0]:.3f}", fontsize=10)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    ax.set_xlim(0.74, 1.16)
    fig.tight_layout()
    os.makedirs("output/figures", exist_ok=True)
    out = "output/figures/llh_cc0cc1_norm_degeneracy.png"
    fig.savefig(out, dpi=140); print(f"wrote {out}", flush=True)
    print(f"CC0pi-only : qe={d0['bfp'][0]:.3f}+/-{s0[0]:.3f}  res={d0['bfp'][1]:.3f}+/-{s0[1]:.3f}")
    print(f"CC0pi+CC1pi: qe={dc['bfp'][0]:.3f}+/-{sc[0]:.3f}  res={dc['bfp'][1]:.3f}+/-{sc[1]:.3f}")


if __name__ == "__main__":
    main()
