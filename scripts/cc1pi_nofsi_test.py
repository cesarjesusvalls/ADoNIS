"""CARBON NO-FSI validation, done with CENTRAL code only.
ADoNIS: res_xsec.generate primary (t-channel default, spline) -> select the T2K CC1pi+Np signal
EXACTLY as ACHILLES does at the primary level: pi+ (ppid==211), a REAL proton (Npid==2212, the RES
nucleon -- no FSI knockouts exist), proton/mu/pi all in acceptance.  STV via the central observables().
ACHILLES: the central no-FSI extraction t2k_cc1pi_tki_achilles_nofsi.npz (extract_t2k_cc1pi_tki.py on
T2K_CH_virt_nofsi.hepmc), carbon = is_h==False.  No bespoke candidate logic, no pid hardcoding.

If the high-W/pi_p tail survives here -> nuclear primary (spectral/de-Forest).  If clean -> the tail
needs the cascade (FSI).

Usage: python scripts/cc1pi_nofsi_test.py [NRES=120000] [NSEED=4] [--recompute]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.xsec import res_xsec
# central observables + acceptance + signal windows (importing pins spline via dcc)
from scripts.cc1pi_fig_tki import observables, _acc, MU_LO, MU_HI, PI_LO, PI_HI, P_LO, P_HI
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")
VARS = [("pn", np.array([0, 120, 240, 600, 1500.0]), r"$p_N$ [MeV/c]"),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.0]), r"$\delta p_{TT}$"),
        ("dalphat", np.linspace(0, np.pi, 7), r"$\delta\alpha_T$ [rad]"),
        ("W", np.linspace(1080, 1700, 13), r"vertex $W$ [MeV]"),
        ("Q2", np.linspace(0, 1.5, 13), r"$Q^2$ [GeV$^2$]"),
        ("pi_p", np.linspace(150, 1200, 13), r"$p_\pi$ [MeV/c]"),
        ("lp_p", np.linspace(450, 1200, 13), r"lead $p_p$ [MeV/c]")]


def ado_one(seed):
    e = res_xsec.generate(NRES, seed=seed, return_events=True)["events"]
    knu, kmu = np.asarray(e["k_nu"]), np.asarray(e["k_mu"])
    pstr, ppi, pN = np.asarray(e["p_struck"]), np.asarray(e["p_pi"]), np.asarray(e["p_N"])
    w = np.asarray(e["w"]); ppid = np.asarray(e["ppid"]); Npid = np.asarray(e["Npid"])
    # T2K CC1pi+Np at the no-FSI primary: pi+, a REAL proton (RES nucleon is the only proton, no
    # knockouts), all in acceptance.  (n->n pi+ has a neutron -> no proton -> excluded, as in ACHILLES.)
    sel = ((ppid == 211) & (Npid == 2212) & (w > 0)
           & _acc(kmu, MU_LO, MU_HI) & _acc(ppi, PI_LO, PI_HI) & _acc(pN, P_LO, P_HI))
    dptt, pn, dat, dpt = observables(kmu[sel], ppi[sel], pN[sel], np.zeros(int(sel.sum()), bool), seed)
    pim = np.linalg.norm(ppi[sel][:, 1:], axis=1)
    qv = (knu - kmu)[sel]; totv = qv + pstr[sel]
    Wv = np.sqrt(np.clip(totv[:, 0] ** 2 - np.sum(totv[:, 1:] ** 2, axis=1), 0, None))
    Q2v = (np.sum(qv[:, 1:] ** 2, axis=1) - qv[:, 0] ** 2) / 1e6
    return dict(dptt=dptt, pn=pn, dalphat=dat, W=Wv, Q2=Q2v, pi_p=pim,
                lp_p=np.linalg.norm(pN[sel][:, 1:], axis=1), w=w[sel])


def get_ado():
    cache = "data/oracle/t2k_cc1pi_nofsi_adonis.npz"
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        d = np.load(cache); print("loaded cached ADoNIS no-FSI", flush=True)
        return {k: d[k] for k in KEYS + ("w",)}
    parts = [ado_one(sd) for sd in range(NSEED)]
    out = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("w",)}
    out["w"] = out["w"] / NSEED
    np.savez(cache, **{k: out[k] for k in KEYS + ("w",)})
    print(f"  ADoNIS no-FSI carbon (p->p pi+): {len(out['w'])} ev  sigma={out['w'].sum():.4e}", flush=True)
    return out


def main():
    ado = get_ado()
    a = np.load("data/oracle/t2k_cc1pi_tki_achilles_nofsi.npz")
    cm = ~np.asarray(a["is_h"])                                  # carbon
    aw = np.asarray(a["w"])[cm] * weight_to_nb_of(a)
    ach = {k: np.asarray(a[k])[cm] for k in ("pn", "dptt", "dalphat", "W", "pi_p", "lp_p")}
    ach["Q2"] = np.asarray(a["Q2"])[cm] / 1e6                    # ACH Q2 MeV^2 -> GeV^2
    print(f"  ACHILLES no-FSI carbon: {int(cm.sum())} ev  sigma={aw.sum():.4e}", flush=True)
    print(f"\n  sigma ACH/ADO = {aw.sum()/max(ado['w'].sum(),1e-30):.3f}")
    fig, axes = plt.subplots(2, len(VARS), figsize=(3.4 * len(VARS), 6.6), height_ratios=[3, 1.2])
    print(f"\n{'var':10s} {'chi2':>8}  {'ACH/ADO':>8}")
    for c, (key, edges, xlab) in enumerate(VARS):
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])

        def H(d, w):
            h, _ = np.histogram(np.asarray(d), bins=edges, weights=np.asarray(w))
            e2, _ = np.histogram(np.asarray(d), bins=edges, weights=np.asarray(w) ** 2)
            return h / bw, np.sqrt(e2) / bw
        da, ea = H(ach[key], aw); dd, ed = H(ado[key], ado["w"])
        ax, axr = axes[0, c], axes[1, c]
        ax.fill_between(edges, np.append(da - ea, (da - ea)[-1]), np.append(da + ea, (da + ea)[-1]),
                        step="post", color="0.45", alpha=0.35, lw=0, label="ACHILLES stat. unc.")
        ax.step(edges, np.append(da, da[-1]), where="post", color="0.35", lw=1.4, label="ACHILLES no-FSI C")
        ax.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=4, capsize=2, lw=1.0,
                    label="ADoNIS C no-FSI (t-chan)", zorder=4)
        ax.set_title(xlab, fontsize=9); ax.set_ylim(bottom=0)
        if c == 0:
            ax.legend(fontsize=6); ax.set_ylabel(r"d$\sigma$/dx [nb/unit]")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = da / dd; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
        m = (da > 0) & (dd > 0) & np.isfinite(re)
        c2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))) / max(int(m.sum()), 1)
        axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green", lw=0.7)
        axr.errorbar(ctr[m], r[m], yerr=re[m], fmt="s", color="C0", ms=3, capsize=2, lw=0.9)
        axr.set_ylim(0.5, 1.6); axr.set_xlabel(xlab, fontsize=8)
        if c == 0:
            axr.set_ylabel("ACH/ADO")
        axr.text(0.04, 0.84, f"{c2:.1f}", transform=axr.transAxes, fontsize=8, color="C0")
        print(f"{key:10s} {c2:8.2f}  {float(np.sum(da*bw))/max(float(np.sum(dd*bw)),1e-30):8.3f}")
    fig.suptitle("T2K CC1$\\pi^+$Np CARBON NO-FSI (central): ADoNIS p->p$\\pi^+$ primary vs ACHILLES no-FSI", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "paper_figures/cc1pi_nofsi.png"; fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
