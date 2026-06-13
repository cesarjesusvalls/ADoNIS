"""The CC1pi+ diagnostic (7-variable ACH/ADO ratios) in a 2x2 matrix:
  (signal-def | no-signal-def) x (with-FSI | no-FSI).
Disentangles what the cascade does vs what the T2K acceptance selection does, per variable.

ADoNIS: res_C/res_H(return_all=True) -> CC1pi+-topology superset (STV computed with the
leading-OVERALL proton for the no-signal-def cell and the leading-ACCEPTED proton for the
signal-def cell).  ACHILLES: scripts/extract_t2k_cc1pi_4way.py topology npz (same superset).
Consistent absolute weighting both sides (ACH w x weight_to_nb; ADO per-event nb / NSEED).

Usage: python scripts/cc1pi_4way.py [NRES=120000] [NSEED=4] [--recompute]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scripts.cc1pi_fig_tki as F
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
NH = 40000
RECOMPUTE = "--recompute" in sys.argv
SKEYS = ("w", "is_h", "mu_p", "mu_cth", "pi_p", "pi_cth", "W", "Q2",
         "dptt_all", "pn_all", "dat_all", "dpt_all", "lp_p_all", "lp_cth_all",
         "has_acc", "dptt_acc", "pn_acc", "dat_acc", "dpt_acc", "lp_p_acc", "mu_acc", "pi_acc")

# variable -> (signal-def key, no-signal-def key, edges, label)
VARS = [
    ("pn",      "pn_acc",   "pn_all",   np.array([0, 120, 240, 600, 1500.0]),        r"$p_N$"),
    ("dptt",    "dptt_acc", "dptt_all", np.array([-700,-300,-100,100,300,700.0]),    r"$\delta p_{TT}$"),
    ("dalphat", "dat_acc",  "dat_all",  np.linspace(0, np.pi, 7),                     r"$\delta\alpha_T$"),
    ("W",       "W",        "W",        np.linspace(1080, 1700, 13),                  r"$W$"),
    ("Q2",      "Q2",       "Q2",       np.linspace(0, 1.5, 13),                      r"$Q^2$"),
    ("pi_p",    "pi_p",     "pi_p",     np.linspace(150, 1200, 13),                   r"$p_\pi$"),
    ("lp_p",    "lp_p_acc", "lp_p_all", np.linspace(0, 1200, 13),                     r"lead $p_p$"),
]
CELLS = ["signal-def", "no-signal-def"]
FSIS = ["FSI", "noFSI"]


def accumulate_ado(fsi):
    parts_c = [F.res_C(NRES, sd, fsi=fsi, return_all=True) for sd in range(NSEED)]
    parts_h = [F.res_H(NH, sd, fsi=fsi, return_all=True) for sd in range(NSEED)]
    out = {}
    for k in SKEYS:
        ac = np.concatenate([p[k] for p in parts_c]); ah = np.concatenate([p[k] for p in parts_h])
        out[k] = np.concatenate([ac, ah])
    out["w"] = out["w"] / NSEED                              # NSEED replicas -> mean
    print(f"  ADoNIS {'FSI' if fsi else 'noFSI'}: N={len(out['w'])} acc={int(out['has_acc'].sum())} "
          f"sumw={out['w'].sum():.4e} nb", flush=True)
    return out


def get_ado():
    cache = "data/oracle/t2k_cc1pi_4way_adonis.npz"
    if os.path.exists(cache) and not RECOMPUTE:
        d = np.load(cache)
        return ({k: d["fsi_" + k] for k in SKEYS}, {k: d["nofsi_" + k] for k in SKEYS})
    af = accumulate_ado(True); an = accumulate_ado(False)
    np.savez(cache, **{"fsi_" + k: af[k] for k in SKEYS}, **{"nofsi_" + k: an[k] for k in SKEYS})
    return af, an


def cell_arrays(src, cell, is_ach):
    """Return {var: (values, weights)} for one cell.  src is an ACH npz or an ADO dict."""
    w = np.asarray(src["w"]).copy()
    if is_ach:
        w = w * weight_to_nb_of(src)
    if cell == "signal-def":
        m = np.asarray(src["has_acc"]) & np.asarray(src["mu_acc"]) & np.asarray(src["pi_acc"])
        suf = "_acc"
    else:
        m = np.ones(len(w), bool); suf = "_all"
    res = {}
    for short, sk, nk, edges, lab in VARS:
        key = sk if cell == "signal-def" else nk
        v = np.asarray(src[key]).astype(float)
        if short == "Q2" and is_ach:
            v = v / 1e6                                     # ACH Q2 in MeV^2 -> GeV^2
        res[short] = (v[m], w[m])
    return res


def main():
    af, an = get_ado()
    ach_f = np.load("data/oracle/t2k_cc1pi_4way_ach_fsi.npz")
    ach_n = np.load("data/oracle/t2k_cc1pi_4way_ach_nofsi.npz")
    ach = {"FSI": ach_f, "noFSI": ach_n}; ado = {"FSI": af, "noFSI": an}

    rows = [(cell, f) for cell in CELLS for f in FSIS]      # 4 cells
    fig, axes = plt.subplots(len(rows), len(VARS), figsize=(2.5 * len(VARS), 2.3 * len(rows)),
                             squeeze=False)
    print(f"\n{'cell':>14} {'fsi':>6} | " + " ".join(f"{v[0]:>8}" for v in VARS) + " | int ACH/ADO")
    for ri, (cell, f) in enumerate(rows):
        A = cell_arrays(ach[f], cell, True); D = cell_arrays(ado[f], cell, False)
        line = f"{cell:>14} {f:>6} | "; iA_tot = iD_tot = 0.0
        for ci, (short, sk, nk, edges, lab) in enumerate(VARS):
            bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
            av, aw = A[short]; dv, dw = D[short]

            def hist(v, ww):
                h, _ = np.histogram(v, bins=edges, weights=ww)
                e2, _ = np.histogram(v, bins=edges, weights=ww ** 2)
                return h / bw, np.sqrt(e2) / bw
            da, ea = hist(av, aw); dd, ed = hist(dv, dw)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
            mok = (da > 0) & (dd > 0) & np.isfinite(re) & (re > 0)
            chi2 = float(np.sum((da[mok] - dd[mok]) ** 2 / (ea[mok] ** 2 + ed[mok] ** 2)))
            ndf = int(mok.sum()); c2n = chi2 / max(ndf, 1)
            ax = axes[ri][ci]
            ax.axhspan(0.9, 1.1, color="green", alpha=0.12); ax.axhline(1.0, ls="--", color="green", lw=0.7)
            ax.errorbar(ctr[mok], r[mok], yerr=re[mok], fmt="o", color="C3", ms=2.5, capsize=1.5, lw=0.8)
            ax.set_ylim(0.4, 1.8)
            ax.text(0.05, 0.84, f"{c2n:.1f}", transform=ax.transAxes, fontsize=8,
                    color="darkred", weight="bold")
            if ri == 0:
                ax.set_title(lab, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{cell}\n{f}\nACH/ADO", fontsize=8)
            iA = float(np.sum(da * bw)); iD = float(np.sum(dd * bw)); iA_tot += iA; iD_tot += iD
            line += f"{c2n:8.1f} "
        line += f"| {iA_tot/max(iD_tot,1e-30):.3f}"
        print(line, flush=True)
    fig.suptitle("T2K CC1$\\pi^+$Np diagnostic — ACH/ADO ratio, 2x2: (signal-def|no-signal-def) x (FSI|noFSI)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = "paper_figures/cc1pi_4way.png"; fig.savefig(out, dpi=120); print("\nwrote", out)


if __name__ == "__main__":
    main()
