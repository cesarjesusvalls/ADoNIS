"""Intuitive 'is any RELEVANT region mismodeled?' diagnostic: one scatter point per phase-space bin,
x = the bin's fractional contribution to the integrated cross section (how much it MATTERS),
y = ACH/ADO deviation (how WRONG it is), colour = significance |pull| (= |ACH-ADO|/sqrt(err^2)).

Bins pooled across several observables (leading-proton |p|, leading-neutron |p|, q0, q3, and the
N(p)/N(n)/N(pi) multiplicities).  A bin in the TOP/BOTTOM-RIGHT (large x, ratio far from 1, red) = a
relevant, significant mismodeling.  Bins far from 1 only at small x (left) are statistically irrelevant.

Run: CHANNEL=qe ACH_BANK=...proc.npz python -u scripts/deviation_vs_contribution.py <out.png>
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dev_vs_contrib.png"
CH = os.environ.get("CHANNEL", "qe").lower()
ADO_DIR = os.environ.get("ADO_DIR", "data/oracle")
ACH = os.environ["ACH_BANK"]
PROC = [200] if CH == "qe" else [401, 402]
ADO_GLOB = f"{ADO_DIR}/t2k_{'cc0pi' if CH=='qe' else 'cc1pi'}_cv5_batch*.npz"
COS70 = float(np.cos(np.deg2rad(70.0)))


def mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > 250) & (p < 7000) & (cz > COS70)


def hist(vals, w, edges):
    h, _ = np.histogram(vals, edges, weights=w); h2, _ = np.histogram(vals, edges, weights=w ** 2)
    return h, np.sqrt(h2)                                   # integrated sigma per bin + its error


def load_ado():
    fs = [f for f in sorted(glob.glob(ADO_GLOB)) if "neut" in np.load(f).files]; nb = len(fs)
    d = {k: [] for k in ("w", "mu", "pl", "nl", "q0", "q3", "n_p", "n_n", "n_pip")}
    for f in fs:
        b = dict(np.load(f, allow_pickle=True)); m = mu_mask(b["mu"]) & (b["w"] > 0)
        d["w"].append(b["w"][m]); d["mu"].append(b["mu"][m])
        d["pl"].append(np.linalg.norm(b["prot"][m][:, 0, 1:], axis=1))
        d["nl"].append(np.linalg.norm(b["neut"][m][:, 0, 1:], axis=1))
        qv = b["nu"][m] - b["mu"][m]; d["q0"].append(qv[:, 0]); d["q3"].append(np.linalg.norm(qv[:, 1:], axis=1))
        for k in ("n_p", "n_n", "n_pip"):
            d[k].append(b[k][m])
    return {k: np.concatenate(v) for k, v in d.items()}, nb


def load_ach():
    d = dict(np.load(ACH, allow_pickle=True)); sel = np.isin(d["proc"], PROC) & mu_mask(d["mu"])
    w = d["w"][sel] * float(d["weight_to_nb"]); mu = d["mu"][sel]
    pl = np.linalg.norm(d["prot_p4"][sel][:, 0, 1:], axis=1)
    nl = np.linalg.norm(d["neut_p4"][sel][:, 0, 1:], axis=1)
    qv = d["nu"][sel] - mu; q0 = qv[:, 0]; q3 = np.linalg.norm(qv[:, 1:], axis=1)
    pip = (d["pi_pid"][sel] == 211).sum(1)
    npr = (np.linalg.norm(d["prot_p4"][sel][:, :, 1:], axis=2) > 1e-6).sum(1)
    nne = (np.linalg.norm(d["neut_p4"][sel][:, :, 1:], axis=2) > 1e-6).sum(1)
    return dict(w=w, pl=pl, nl=nl, q0=q0, q3=q3, n_p=npr, n_n=nne, n_pip=pip)


def main():
    A, nb = load_ado(); H = load_ach(); wa = A["w"] / nb; wh = H["w"]
    sig_tot = wa.sum()
    OBS = [("lead p_p", "pl", np.linspace(0, 1500, 21)), ("lead p_n", "nl", np.linspace(0, 1500, 21)),
           ("q0", "q0", np.linspace(0, 1200, 21)), ("q3", "q3", np.linspace(0, 1200, 21)),
           ("N(p)", "n_p", np.arange(7)), ("N(n)", "n_n", np.arange(7)), ("N(pi+)", "n_pip", np.arange(4))]
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.viridis
    for name, key, edges in OBS:
        da, ea = hist(A[key], wa, edges); dh, eh = hist(H[key], wh, edges)
        m = (da > 0) & (dh > 0)
        frac = da[m] / sig_tot                              # bin contribution to integrated sigma
        ratio = dh[m] / da[m]
        pull = np.abs(dh[m] - da[m]) / np.sqrt(ea[m] ** 2 + eh[m] ** 2 + 1e-300)
        rerr = ratio * np.sqrt((ea[m] / da[m]) ** 2 + (eh[m] / dh[m]) ** 2)
        sc = ax.scatter(frac, ratio, c=np.clip(pull, 0, 5), cmap=cmap, vmin=0, vmax=5,
                        s=34, edgecolor="k", lw=0.3, zorder=3)
        ax.errorbar(frac, ratio, yerr=rerr, fmt="none", ecolor="0.6", lw=0.5, zorder=1)
        # label the relevant outliers: contribute >1% AND |ratio-1|>5%
        for x, y, p in zip(frac, ratio, pull):
            if x > 0.01 and abs(y - 1) > 0.05 and p > 2:
                ax.annotate(name, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(1.0, color="green", lw=1); ax.axhspan(0.95, 1.05, color="green", alpha=0.10)
    ax.axvline(0.01, color="0.7", ls=":", lw=1); ax.text(0.0102, ax.get_ylim()[0], " 1% of σ", fontsize=8, color="0.4")
    ax.set_xscale("log"); ax.set_xlabel("bin contribution to integrated σ  (dσ_bin / σ_total)")
    ax.set_ylabel("ACH / ADoNIS  (deviation)"); ax.set_ylim(0.5, 1.6)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("significance |pull| = |ACH−ADO| / σ_stat (capped 5)")
    ax.set_title(f"C-{CH.upper()}: per-bin deviation vs σ-contribution (mass-fixed, Gaussian).  "
                 f"Danger = right + far-from-1 + red.")
    fig.tight_layout(); fig.savefig(OUT, dpi=120); print(f"wrote {OUT}", flush=True)
    # text summary: the sigma-weighted mean |deviation|, and the worst relevant bins
    print(f"  channel C-{CH.upper()}  sigma_tot ADoNIS={sig_tot:.4e}", flush=True)


if __name__ == "__main__":
    main()
