"""NO-FSI inclusive-pi+ test: is the n->n pi+ PRIMARY pi+ over-produced at high W, or clean?
Selection = T2K signal acceptance MINUS the proton requirement (mu in [250,7000]&theta<70,
pi+ in [150,1200]&theta<70, NO proton).  p->p pi+ is already validated clean (cc1pi_nofsi_test),
so any high-W/pi_p tail in the inclusive pi+ is the n->n pi+ primary (which has no primary proton
and only enters the real signal via an FSI knockout proton).

ADoNIS: res_xsec.generate primary (t-channel, spline), inclusive pi+ (tagged by Npid for display).
ACHILLES: t2k_res_w_achilles_nofsi.npz (no-FSI broad RES), pi_pid==211, same acceptance; w x weight_to_nb
from the no-FSI hepmc header.

If clean -> the CH-FSI tail is the cascade (knockout) path.  If tail present -> n->n pi+ primary.
Usage: python scripts/cc1pi_nofsi_pip.py [NRES=120000] [NSEED=4] [--recompute]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.xsec import res_xsec
from scripts.cc1pi_fig_tki import _acc, MU_LO, MU_HI, PI_LO, PI_HI   # central acceptance + windows
from adonis.data.oracle.normalization import hepmc_norm

NRES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
COS70 = np.cos(np.deg2rad(70.0))
NOFSI_HEPMC = "_oracle_out/T2K_CH_virt_nofsi.hepmc"
WED = np.linspace(1080, 1700, 13); PED = np.linspace(150, 1200, 13)


def ado_one(seed):
    e = res_xsec.generate(NRES, seed=seed, return_events=True)["events"]
    knu, kmu = np.asarray(e["k_nu"]), np.asarray(e["k_mu"])
    pstr, ppi = np.asarray(e["p_struck"]), np.asarray(e["p_pi"])
    w = np.asarray(e["w"]); ppid = np.asarray(e["ppid"]); Npid = np.asarray(e["Npid"])
    sel = ((ppid == 211) & (w > 0) & _acc(kmu, MU_LO, MU_HI) & _acc(ppi, PI_LO, PI_HI))  # NO proton req
    qv = (knu - kmu)[sel]; totv = qv + pstr[sel]
    W = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    pip = np.linalg.norm(ppi[sel][:, 1:], axis=1)
    return dict(W=W, pi_p=pip, w=w[sel], nnpi=(Npid[sel] == 2112))   # nnpi = n->n pi+ tag


def get_ado():
    cache = "data/oracle/t2k_cc1pi_nofsi_pip_adonis.npz"
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        d = np.load(cache); print("loaded cached ADoNIS", flush=True)
        return {k: d[k] for k in ("W", "pi_p", "w", "nnpi")}
    parts = [ado_one(sd) for sd in range(NSEED)]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("W", "pi_p", "w", "nnpi")}
    out["w"] = out["w"] / NSEED
    np.savez(cache, **out)
    f = out["w"][out["nnpi"]].sum() / out["w"].sum()
    print(f"  ADoNIS no-FSI inclusive pi+: {len(out['w'])} ev  sigma={out['w'].sum():.4e}  "
          f"n->npi+ frac={f:.2f}", flush=True)
    return out


def main():
    ado = get_ado()
    a = np.load("data/oracle/t2k_res_w_achilles_nofsi.npz")
    w2nb = hepmc_norm(NOFSI_HEPMC)["weight_to_nb"]
    m = ((np.asarray(a["pi_pid"]) == 211) & (np.asarray(a["mu_p"]) > MU_LO) & (np.asarray(a["mu_p"]) < MU_HI)
         & (np.asarray(a["mu_cth"]) > COS70) & (np.asarray(a["pi_p"]) > PI_LO) & (np.asarray(a["pi_p"]) < PI_HI)
         & (np.asarray(a["pi_cth"]) > COS70))
    aW, aP, aw = np.asarray(a["W"])[m], np.asarray(a["pi_p"])[m], np.asarray(a["w"])[m] * w2nb
    print(f"  ACHILLES no-FSI inclusive pi+: {int(m.sum())} ev  sigma={aw.sum():.4e}  (w2nb={w2nb:.3e})", flush=True)
    print(f"\n  sigma ACH/ADO (inclusive pi+, no-FSI) = {aw.sum()/max(ado['w'].sum(),1e-30):.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), height_ratios=[3, 1.2])
    for c, (key, av, edges, xlab) in enumerate([("W", aW, WED, "vertex W [MeV]"),
                                                 ("pi_p", aP, PED, r"$p_\pi$ [MeV/c]")]):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        dv = ado[key]

        def H(v, w):
            h, _ = np.histogram(np.asarray(v), bins=edges, weights=np.asarray(w))
            e2, _ = np.histogram(np.asarray(v), bins=edges, weights=np.asarray(w) ** 2)
            return h / bw, np.sqrt(e2) / bw
        da, ea = H(av, aw); dd, ed = H(dv, ado["w"])
        dnn, _ = H(dv[ado["nnpi"]], ado["w"][ado["nnpi"]])      # ADoNIS n->n pi+ component
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.45", alpha=0.35, lw=0, label="ACHILLES stat. unc.")
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES no-FSI")
        ax.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=4, capsize=2, lw=1.0,
                    label="ADoNIS no-FSI (incl.)", zorder=4)
        ax.step(edges, np.append(dnn, dnn[-1]), where="post", color="C3", lw=1.0, ls="--",
                label=r"ADoNIS n$\to$n$\pi^+$ part")
        ax.set_title(f"no-FSI inclusive $\\pi^+$  {xlab}", fontsize=10); ax.set_ylim(bottom=0)
        ax.set_ylabel(r"d$\sigma$/dx [nb/unit]"); ax.legend(fontsize=7)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        msk = (da > 0) & (dd > 0) & np.isfinite(re)
        c2 = float(np.sum((da[msk] - dd[msk]) ** 2 / (ea[msk] ** 2 + ed[msk] ** 2))) / max(int(msk.sum()), 1)
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.7)
        axr.errorbar(ctr[msk], r[msk], yerr=re[msk], fmt="s", color="C0", ms=3, capsize=2, lw=0.9)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab); axr.set_ylabel("ACH/ADO")
        axr.text(0.04, 0.84, f"chi2/ndf {c2:.1f}", transform=axr.transAxes, fontsize=9)
        print(f"  {key:6s}: chi2/ndf={c2:.2f}  integral ACH/ADO={float(np.sum(da*bw))/max(float(np.sum(dd*bw)),1e-30):.3f}")
    fig.suptitle("T2K no-FSI inclusive $\\pi^+$ (signal acceptance, NO proton req): ADoNIS vs ACHILLES", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "paper_figures/cc1pi_nofsi_pip.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
