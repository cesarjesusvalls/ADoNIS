"""MEC-admixture verdict figure — instant, from the persisted run npz (logbook 19).

4 CC0pi overlays (data = ADoNIS nominal + GENIE MEC; nominal; M0 biased fit; M1 clean-region fit)
with the M1-excised regions shaded, + knob-pull comparison + chi2 summary."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physical_fit import NPAR

C_M0, C_M1, C_M2, C_NOM = "#d62728", "#1f77b4", "#ff7f0e", "0.45"
z = np.load("output/altgen/p9_mecmix.npz", allow_pickle=True)
pn = list(z["pnames"]); truth = np.asarray(z["truth"])
row0 = np.asarray(z["row0"]); keys = list(z["dskeys"])
data = np.asarray(z["data"]); sigma = np.asarray(z["sigma"]); m_nom = np.asarray(z["model_nom"])

def flags_at_nominal(j):
    s = slice(row0[j], row0[j+1])
    pull = (data[s] - m_nom[s]) / sigma[s]
    bad = np.abs(pull) > 2.0
    runs, i = [], 0
    while i < len(bad):
        if bad[i]:
            k = i
            while k + 1 < len(bad) and bad[k + 1]:
                k += 1
            if k - i + 1 >= 2:
                runs.append((i, k))
            i = k + 1
        else:
            i += 1
    return runs

XLAB = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]",
        "pmu": r"$p_\mu$ [MeV/c]", "cosmu": r"$\cos\theta_\mu$"}
fig = plt.figure(figsize=(13.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
panels = ["dpt", "dat", "pmu", "cosmu"]
for i, key in enumerate(panels):
    j = keys.index(key); s = slice(row0[j], row0[j+1])
    edges = np.asarray(z[f"{key}_edges"])
    x = 0.5 * (edges[1:] + edges[:-1]); xe = 0.5 * np.diff(edges)
    ax = fig.add_subplot(gs[i // 2, i % 2])
    for (i0, i1) in flags_at_nominal(j):
        ax.axvspan(edges[i0], edges[i1 + 1], color="#f0e0e0", zorder=0)
    ax.errorbar(x, data[s], xerr=xe, yerr=sigma[s], fmt="o", color="k", ms=3, lw=0.7, capsize=0,
                zorder=3, label="ADoNIS nominal + GENIE MEC")
    xs = np.repeat(edges, 2)[1:-1]
    ax.plot(xs, np.repeat(m_nom[s], 2), color=C_NOM, ls=":", lw=1.5, label="ADoNIS nominal (truth)")
    ax.plot(xs, np.repeat(np.asarray(z["M0_model"])[s], 2), color=C_M0, lw=1.4, label="M0 biased fit")
    ax.plot(xs, np.repeat(np.asarray(z["M1_model"])[s], 2), color=C_M1, lw=1.5,
            label="M1 clean-region fit")
    ax.set_xlabel(XLAB[key], fontsize=9); ax.set_title(f"CC0pi {key}", fontsize=10)
    ax.set_ylim(bottom=0)
    if i == 0:
        ax.legend(fontsize=7.5)

# knob pulls
axp = fig.add_subplot(gs[0, 2])
knobs = [k for k in [int(x) for x in z["M0_sub"]] if k < NPAR]
yy = np.arange(len(knobs))
for key, lab, col, mk, dy in [("M0", "M0 standard", C_M0, "s", -0.22), ("M2", "M2 Huber", C_M2, "^", 0.0),
                              ("M1", "M1 physical", C_M1, "o", 0.22)]:
    subz = [int(x) for x in z[f"{key}_sub"]]
    V = np.asarray(z[f"{key}_V"]); th = np.asarray(z[f"{key}_th"])
    pulls = []
    for k in knobs:
        if k in subz:
            c = subz.index(k); sig_ = np.sqrt(max(V[c, c], 1e-24))
            pulls.append(np.clip((th[k] - truth[k]) / sig_, -8.5, 8.5))
        else:
            pulls.append(0.0)
    axp.scatter(pulls, yy + dy, color=col, marker=mk, s=30, label=lab, zorder=3)
axp.axvline(0, color="k", lw=1.0); axp.axvspan(-2, 2, color="0.94", zorder=0)
axp.set_yticks(yy); axp.set_yticklabels([pn[k] for k in knobs], fontsize=8)
axp.set_xlabel(r"(BFP $-$ truth) / $\sigma$  (clipped ±8.5)")
axp.set_title("pulls: M0 −23σ ... M1 ≤1.1σ", fontsize=9)
axp.legend(fontsize=7.5); axp.invert_yaxis()

# summary text
axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
axt.text(0, 0.95, ("OUTSIDE-MANIFOLD unknown-unknown\n"
                   "(2p2h: no such channel in ADoNIS)\n\n"
                   f"chi2_data:  M0 {float(z['M0_chi2data']):.0f} | M2 {float(z['M2_chi2data']):.0f} | "
                   f"M1 {float(z['M1_chi2data']):.1f} (clean)\n\n"
                   "M1 excised (shaded): dpt tail, all dat,\n"
                   "3 pmu segments, forward cosmu —\n"
                   "the 2p2h habitat, mapped.\n"
                   "CC1pi untouched (no-pion control).\n\n"
                   "Taxonomy confirmed:\n"
                   "outside-manifold -> coherence gates win;\n"
                   "inside-manifold (Q2 mod) -> habitat basis."),
         fontsize=9.5, va="top", family="DejaVu Sans")
for ax in fig.axes:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle("Physical unknown-unknown: ADoNIS nominal + GENIE 2p2h — who survives?", y=0.99)
out = "output/figures/physfit_fig10_mecmix.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"[fig] {out}")
