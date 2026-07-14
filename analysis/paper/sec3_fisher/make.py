"""Paper section 3 -- Fisher information per observable subset: what is worth fitting, and why not.

Gate I asks, for a given observable subset: with a prior on every knob, does the DATA (not the prior)
determine it?  Asimov posterior V = (J^T C^-1 J + Pi^-1)^-1; a knob is FIT when
    shrinkage = sigma_post / prior < 0.5     (the data at least halves the prior width).

sigma_post is MARGINALIZED (a diagonal element of V), so a knob fails either because the data cannot
SEE it or because another knob can MIMIC it.  Those are physically opposite and want opposite fixes, so
we report both axes:
    raw  = 1/sqrt(F_kk) / prior   -- other knobs held FIXED: pure sensitivity, degeneracy-blind
    marg = sqrt(V_kk)  / prior    -- other knobs free: what the fit actually delivers
  raw < 0.5, marg < 0.5  -> MEASURABLE
  raw < 0.5, marg > 0.5  -> DEGENERATE (seen clearly, cannot be disentangled -- a better observable can fix it)
  raw > 0.5              -> INVISIBLE  (the sample carries no information at this precision -- nothing can)

Everything is a row slice of ONE persisted Jacobian (physfit_gate1_full.npz: 27 knobs x all datasets),
so every subset is exact and costs no bank pass.

Usage:  python -m analysis.paper.sec3_fisher.make [label]      (default label: physfit_gate1_full)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "altgen"))
import physical_fit as PF                      # OBS_SUBSETS = the single source of truth
from analysis.paper import style

FIT_CUT = 0.5
# subsets to show, in the order they answer "what does each class of observable buy you?"
SHOW = ["lepton", "leptonhad", "tki", "mult", "kin9", "full"]
NICE = {"lepton": "lepton\n$p_\\mu,\\cos\\theta_\\mu$", "leptonhad": "lepton\n+hadron",
        "tki": "TKI", "mult": "multipl.\n$N_p,N_{\\pi^\\pm}$", "kin9": "all kin.\n(9 obs)",
        "full": "ALL\n(11 obs)"}


def gate1(J, sigma, prior, rows):
    """(marginalized shrinkage, raw shrinkage, Fisher) for the given bin rows."""
    Jw = J[rows] / sigma[rows, None]
    F = Jw.T @ Jw
    V = np.linalg.inv(F + np.diag(1.0 / prior**2))
    marg = np.sqrt(np.diag(V)) / prior
    with np.errstate(divide="ignore"):
        raw = np.where(np.diag(F) > 0, 1.0 / np.sqrt(np.maximum(np.diag(F), 1e-300)) / prior, np.inf)
    return marg, raw, F


def main(label="physfit_gate1_full"):
    style.use()
    d = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    J, sigma, prior = d["J"], d["sigma"], d["prior"]
    pnames = [str(x) for x in d["pnames"]]
    dskeys = [str(x) for x in d["dskeys"]]
    row0 = np.asarray(d["row0"])
    rows_of = {k: np.arange(row0[j], row0[j + 1]) for j, k in enumerate(dskeys)}

    subsets = [s for s in SHOW if set(PF.OBS_SUBSETS[s]) <= set(dskeys)]
    missing = [s for s in SHOW if s not in subsets]
    if missing:
        print(f"[warn] npz lacks the observables for: {missing} (has {dskeys})")

    M = np.zeros((len(pnames), len(subsets)))          # marginalized shrinkage
    R = np.zeros_like(M)                               # raw shrinkage
    for c, s in enumerate(subsets):
        rows = np.concatenate([rows_of[k] for k in PF.OBS_SUBSETS[s]])
        M[:, c], R[:, c], _ = gate1(J, sigma, prior, rows)

    # ---- stdout table ------------------------------------------------------------------------------ #
    print(f"\n==== GATE I per observable subset ({label}) -- shrinkage = sigma_post/prior, FIT < {FIT_CUT} ====")
    print(f"{'knob':>20} " + " ".join(f"{s:>10}" for s in subsets))
    for k in np.argsort(M[:, -1]):
        print(f"{pnames[k]:>20} " + " ".join(f"{M[k,c]:10.2f}" for c in range(len(subsets))))
    for c, s in enumerate(subsets):
        print(f"  {s:>10}: {int((M[:,c] < FIT_CUT).sum()):2d}/{len(pnames)} FIT  "
              f"[{', '.join(pnames[k] for k in np.where(M[:,c] < FIT_CUT)[0])}]")

    # ---- Fig 1: knob x subset shrinkage, marginalized vs raw ---------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 8.2), sharey=True)
    panels = ((axes[0], M, "MARGINALIZED (other knobs free)\nwhat the fit delivers", "#d62728"),
              (axes[1], R, "RAW (other knobs fixed)\nwhat the data can see", "#2ca02c"))
    for ax, Z, ttl, box in panels:
        im = ax.imshow(np.clip(Z, 0, 1.2), aspect="auto", cmap="Blues", vmin=0, vmax=1.2)
        ax.set_xticks(range(len(subsets)))
        ax.set_xticklabels([NICE.get(s, s) for s in subsets], fontsize=7)
        ax.set_title(ttl, fontsize=8.5)
        for k in range(len(pnames)):
            for c in range(len(subsets)):
                v = Z[k, c]                                   # UNCLIPPED: >1.2 and inf must read ">1"
                txt = "$>$1" if (not np.isfinite(v)) or v > 1.2 else f"{v:.2f}"
                ax.text(c, k, txt, ha="center", va="center", fontsize=6,
                        color="w" if min(v, 1.2) > 0.7 else "k")
                if v < FIT_CUT:
                    ax.add_patch(Rectangle((c - .5, k - .5), 1, 1, fill=False, ec=box, lw=1.4))
    axes[0].set_yticks(range(len(pnames)))
    axes[0].set_yticklabels(pnames, fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="shrinkage  $\\sigma_{post}/\\sigma_{prior}$")
    # the two panels differ ONLY by marginalization: a cell green-but-not-red is a DEGENERATE knob.
    fig.suptitle("Gate I per observable class.   red box = FIT (the fit measures it) · "
                 "green box = the data sees it with other knobs held fixed\n"
                 "green without red  $\\Rightarrow$  DEGENERATE (sensitivity exists, another knob spends it)",
                 fontsize=9)
    style.save(fig, "sec3_shrinkage_subsets")

    # ---- Fig 2: the two failure modes --------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    c_full = subsets.index("full") if "full" in subsets else len(subsets) - 1
    raw, marg = R[:, c_full], M[:, c_full]
    rp = np.clip(raw, 1e-3, 3.0)
    for k in range(len(pnames)):
        meas = marg[k] < FIT_CUT
        deg = (not meas) and raw[k] < FIT_CUT
        col = "#2ca02c" if meas else ("#ff7f0e" if deg else "#7f7f7f")
        ax.scatter(rp[k], np.clip(marg[k], 0, 1.05), s=34, color=col, zorder=3)
        ax.annotate(pnames[k], (rp[k], np.clip(marg[k], 0, 1.05)), fontsize=6,
                    xytext=(4, 2), textcoords="offset points", color=col)
    ax.axvline(FIT_CUT, color="k", ls=":", lw=.8)
    ax.axhline(FIT_CUT, color="k", ls=":", lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("raw shrinkage  (other knobs FIXED)  —  can the data see it?")
    ax.set_ylabel("marginalized shrinkage  —  can the fit deliver it?")
    ax.set_title("Two ways to fail Gate I", fontsize=10)
    # quadrants: x = can the data see it, y = does the fit deliver it (LOW = good on both axes)
    ax.text(.03, .06, "MEASURABLE\nseen, and no other knob can spend it", color="#2ca02c",
            transform=ax.transAxes, fontsize=8, weight="bold")
    ax.text(.03, .93, "DEGENERATE\nseen, but mimicked by another knob\n(a better observable can recover it)",
            color="#ff7f0e", transform=ax.transAxes, fontsize=8, weight="bold", va="top")
    ax.text(.99, .93, "INVISIBLE\nno information at this precision\n(nothing can recover it)",
            color="#555555", transform=ax.transAxes, fontsize=8, weight="bold", va="top", ha="right")
    style.save(fig, "sec3_failure_modes")

    # ---- Fig 3: degeneracy structure (prior-scaled Fisher eigen-spectrum, full set) ------------------ #
    rows = np.concatenate([rows_of[k] for k in PF.OBS_SUBSETS[subsets[c_full]]])
    _, _, F = gate1(J, sigma, prior, rows)
    Fs = F * prior[:, None] * prior[None, :]
    evals, evecs = np.linalg.eigh(Fs)
    order = np.argsort(evals)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1, 2.4]})
    axes[0].semilogy(range(1, len(evals) + 1), evals[order], "o-", ms=4, color=style.C_ADONIS)
    axes[0].axhline(1.0, color="k", ls=":", lw=.8)
    axes[0].text(len(evals) * .55, 1.3, "$\\lambda<1$: prior-dominated", fontsize=7)
    axes[0].set_xlabel("eigenmode"); axes[0].set_ylabel("$\\lambda$  (prior-scaled Fisher)")
    axes[0].set_title("degeneracy spectrum", fontsize=9)
    K = 8
    Wv = evecs[:, order[:K]]
    im = axes[1].imshow(Wv, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels([f"{evals[order[i]]:.3g}" for i in range(K)], fontsize=7, rotation=45)
    axes[1].set_xlabel("$\\lambda$ of the mode")
    axes[1].set_yticks(range(len(pnames))); axes[1].set_yticklabels(pnames, fontsize=7)
    axes[1].set_title("eigenvector composition (best-measured modes)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02, label="knob weight in the mode")
    style.save(fig, "sec3_degeneracy")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
