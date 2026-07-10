"""5-parameter closure figure — reads ONLY the persisted run npz (no bank, instant).

Left: per-knob recovery (injected truth vs M0/M2 BFP with errors, native units).
Right: pulls (BFP - truth)/sigma. Bottom: dpt + dat overlays (data = exact-reweight model at the
injected point; nominal; fitted curves) from the persisted binned curves."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physical_fit import PNAMES

C_M0, C_M2, C_NOM = "#d62728", "#ff7f0e", "0.45"
z = np.load("output/altgen/physfit_closure5.npz", allow_pickle=True)
sub = [int(k) for k in z["subset"]]
truth = np.asarray(z["truth"])
row0 = np.asarray(z["row0"]); dskeys = list(z["dskeys"])
data = np.asarray(z["data"]); sigma = np.asarray(z["sigma"])
m_nom = np.asarray(z["model_nom"])

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

# (a) recovery in native units
ax = fig.add_subplot(gs[0, 0])
yy = np.arange(len(sub))
ax.scatter([truth[k] for k in sub], yy, marker="*", s=140, color="k", zorder=4, label="injected truth")
for key, col, mk, dy in [("M0", C_M0, "s", -0.15), ("M2", C_M2, "^", 0.15)]:
    th = np.asarray(z[f"{key}_th"]); V = np.asarray(z[f"{key}_V"])
    sig = np.sqrt(np.abs(np.diag(V)))
    ax.errorbar([th[k] for k in sub], yy + dy, xerr=sig, fmt=mk, color=col, ms=6, capsize=3,
                lw=1.4, label=f"{key} BFP", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([PNAMES[k] for k in sub])
ax.axvline(1.0, color="0.85", lw=0.8, zorder=0)
ax.set_xlabel("knob value (native units)"); ax.invert_yaxis()
ax.set_title("recovery: injected vs fitted"); ax.legend(fontsize=8)

# (b) pulls
ax = fig.add_subplot(gs[0, 1])
for key, col, mk, dy in [("M0", C_M0, "s", -0.15), ("M2", C_M2, "^", 0.15)]:
    th = np.asarray(z[f"{key}_th"]); V = np.asarray(z[f"{key}_V"])
    sig = np.sqrt(np.abs(np.diag(V)))
    pulls = [(th[k] - truth[k]) / max(sig[c], 1e-12) for c, k in enumerate(sub)]
    ax.errorbar(pulls, yy + dy, xerr=1.0, fmt=mk, color=col, ms=6, capsize=3, lw=1.4, label=key)
ax.axvline(0, color="k", lw=1.0); ax.axvspan(-2, 2, color="0.94", zorder=0)
ax.set_yticks(yy); ax.set_yticklabels([PNAMES[k] for k in sub])
ax.set_xlabel(r"(BFP $-$ injected) / $\sigma$"); ax.set_xlim(-4, 4)
ax.set_title("pulls (band = ±2σ)"); ax.legend(fontsize=8); ax.invert_yaxis()

# (c,d) overlays: dpt + dat from persisted curves
XL = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]"}
for col_i, want in enumerate(["dpt", "dat"]):
    j = dskeys.index(want); s = slice(row0[j], row0[j + 1])
    edges = np.asarray(z[f"{want}_edges"])
    x = 0.5 * (edges[1:] + edges[:-1]); xe = 0.5 * np.diff(edges)
    ax = fig.add_subplot(gs[1, col_i])
    ax.errorbar(x, data[s], xerr=xe, yerr=sigma[s], fmt="o", color="k", ms=3, lw=0.8,
                capsize=0, zorder=3, label="closure data (model @ injected θ)")
    xs = np.repeat(edges, 2)[1:-1]
    ax.plot(xs, np.repeat(m_nom[s], 2), color=C_NOM, ls=":", lw=1.6, label="ADoNIS nominal")
    for key, col, ls in [("M0", C_M0, "-"), ("M2", C_M2, "--")]:
        ax.plot(xs, np.repeat(np.asarray(z[f"{key}_model"])[s], 2), color=col, ls=ls, lw=1.5,
                label=f"{key} BFP")
    ax.set_xlabel(XL[want]); ax.set_ylabel(r"$d\sigma/dx$"); ax.set_ylim(bottom=0)
    ax.set_title(f"CC0pi {want}")
    if col_i == 0:
        ax.legend(fontsize=8)
for ax in fig.axes:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle("5-parameter closure: inject M_A_res=0.85, kF_sf=1.10, Eb=3 MeV, s_NN_el[pn]=1.25, "
             "f_NN_cex=0.40 — fit blind (M0, M2)", y=0.98)
out = "output/figures/physfit_fig7_closure5.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"[fig] {out}")
