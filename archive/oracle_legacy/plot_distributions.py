"""Plot the oracle differential distributions parsed from the ACHILLES events.
Run after parse_hepmc.py.  Produces oracle/oracle_distributions.png."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load(Path(__file__).resolve().parent / "oracle_distributions.npz")


def step(ax, edges, vals, **kw):
    c = 0.5 * (edges[:-1] + edges[1:])
    ax.step(c, vals, where="mid", **kw)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
step(axes[0, 0], D["Q2_edges"] / 1e6, D["dsig_Q2"] * 1e6, color="C0")
axes[0, 0].set_xlabel(r"$Q^2$ [GeV$^2$]"); axes[0, 0].set_ylabel(r"$d\sigma/dQ^2$ [nb/GeV$^2$]")
axes[0, 0].set_title("oracle  $d\\sigma/dQ^2$")

step(axes[0, 1], D["W_edges"], D["dsig_W"], color="C1")
axes[0, 1].set_xlabel(r"$W$ [MeV]"); axes[0, 1].set_ylabel(r"$d\sigma/dW$ [nb/MeV]")
axes[0, 1].set_title("oracle  $d\\sigma/dW$  (Delta peak ~1232)")
axes[0, 1].axvline(1232.0, ls="--", color="k", lw=0.8, alpha=0.6)

step(axes[1, 0], D["ppi_edges"], D["dsig_ppi"], color="C2")
axes[1, 0].set_xlabel(r"$|p_\pi|$ [MeV]"); axes[1, 0].set_ylabel(r"$d\sigma/d|p_\pi|$ [nb/MeV]")
axes[1, 0].set_title("oracle  pion-momentum spectrum")

step(axes[1, 1], D["cth_edges"], D["dsig_cth"], color="C3")
axes[1, 1].set_xlabel(r"$\cos\theta_\pi$ (w.r.t. $\vec q$)"); axes[1, 1].set_ylabel(r"$d\sigma/d\cos\theta$ [nb]")
axes[1, 1].set_title("oracle  pion angular distribution")

fig.suptitle(f"ACHILLES single-pion oracle  (n_signal={int(D['n_signal'])}, "
             f"total={float(D['ref_total_nb']):.1f} nb)", fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parent / "oracle_distributions.png"
fig.savefig(out, dpi=110)
print(f"saved -> {out}")
