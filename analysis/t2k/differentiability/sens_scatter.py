"""Knob-sensitivity scatter: one point per knob at (sensitivity to dpt, sensitivity to dat).

Sensitivity = the gradient norm in the DATA covariance metric, s_o^k = sqrt( J_o^k . Sigma_o^-1 . J_o^k ),
i.e. "how many sigma the prediction on observable o moves per unit of knob k" (J = per-bin Jacobian from
grad_arrows' saved npz; Sigma = the T2K STV covariance).  A point high&right is well constrained by BOTH
observables; a point hugging one axis is informative only for that observable.

  python analysis/t2k/differentiability/sens_scatter.py

NOTE: each axis uses the SAME knob unit, so the dpt-vs-dat comparison per knob is exact; cross-knob
comparison inherits each knob's own unit (e.g. Eb_shift is per-MeV) -- read it as relative, not absolute.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

RUN = "/tmp/adonis_tune_runs"
GROUP = {  # label-prefix -> (mechanism, color)
    "hard": ("hard vertex", "#c0392b"), "fsi": ("FSI", "#2471a3"),
    "sf": ("spectral fn", "#27ae60"), "norm": ("normalization", "#8e44ad")}
HARD = {"M_A", "axial_strength", "vector_strength", "mu_p", "mu_n", "gep", "gen",
        "res_axial_strength", "pion_pole"}
SF = {"kF_sf", "Eb_shift", "sf_norm", "src_tail"}
NORM = {"qe_norm", "res_norm"}


def _mech(lab):
    base = lab.split("[")[0]
    if base in HARD: return "hard"
    if base in SF: return "sf"
    if base in NORM: return "norm"
    return "fsi"


def _covinv(obs):
    import uproot
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if obs == 'dpt' else 'dat'}Results.root")
    cov = np.asarray(r["Covariance_Matrix"].values())
    return np.linalg.inv(cov + 1e-12 * np.eye(cov.shape[0]))


def main():
    d_dat = np.load(f"{RUN}/jac_grid_dat.npz", allow_pickle=True)
    d_dpt = np.load(f"{RUN}/jac_grid_dpt.npz", allow_pickle=True)
    labels = list(d_dat["labels"])
    Jdat, Jdpt = np.asarray(d_dat["J"]), np.asarray(d_dpt["J"])     # (nknob, nbins)
    Cidat, Cidpt = _covinv("dat"), _covinv("dpt")
    s_dat = np.sqrt(np.clip(np.einsum("ki,ij,kj->k", Jdat, Cidat, Jdat), 0, None))
    s_dpt = np.sqrt(np.clip(np.einsum("ki,ij,kj->k", Jdpt, Cidpt, Jdpt), 0, None))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fl = 1e-4 * max(s_dat.max(), s_dpt.max())                       # floor so zeros are visible on log axes
    x = np.clip(s_dpt, fl, None); y = np.clip(s_dat, fl, None)
    fig, ax = plt.subplots(figsize=(9, 8))
    lim = (fl * 0.6, max(x.max(), y.max()) * 1.8)
    ax.plot(lim, lim, "--", color="0.6", lw=1, zorder=0, label="equal sensitivity")
    seen = set()
    for lab, xi, yi in zip(labels, x, y):
        g = _mech(lab); name, col = GROUP[g]
        ax.scatter(xi, yi, s=55, color=col, edgecolor="k", lw=0.5, zorder=3,
                   label=name if name not in seen else None)
        seen.add(name)
        ax.annotate(lab, (xi, yi), fontsize=7, xytext=(4, 3), textcoords="offset points", zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(r"sensitivity on $\delta p_T$:  $\sqrt{J^{\mathrm{T}}\Sigma^{-1}J}$  [$\sigma$ / unit knob]")
    ax.set_ylabel(r"sensitivity on $\delta\alpha_T$:  $\sqrt{J^{\mathrm{T}}\Sigma^{-1}J}$  [$\sigma$ / unit knob]")
    ax.set_title("Knob sensitivity: $\\delta p_T$ vs $\\delta\\alpha_T$  (covariance-weighted gradient norm)")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25, which="both")
    os.makedirs("output/figures", exist_ok=True)
    FIG = "output/figures/cc0pi_knob_sensitivity_scatter.png"
    fig.tight_layout(); fig.savefig(FIG, dpi=130); print(f"wrote {FIG}", flush=True)
    order = np.argsort(-(s_dat + s_dpt))
    for k in order:
        print(f"  {labels[k]:>16s}  dpt={s_dpt[k]:.3e}  dat={s_dat[k]:.3e}", flush=True)


if __name__ == "__main__":
    main()
