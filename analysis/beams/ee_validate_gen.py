"""Material- and channel-general (e,e') validation: merge ADoNIS shards + compare to the ACHILLES oracle
dsigma/domega, for any (channel in {qe,res}, material in {C,Ar}).  Reads shards from
output/ee_{mat}_{ch}_shards/part_*.npz and the oracle from output/oracle_ee_{mat}_{ch}.

Usage:  python -m analysis.beams.ee_validate_gen <qe|res> <C|Ar>
"""
from __future__ import annotations
import glob, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.utils.hepmc import parse_events, hepmc_norm   # noqa: E402

E_BEAM = 2222.0
THETA_ACC = (14.0, 17.0)
NUC = {"C": "^{12}C", "Ar": "^{40}Ar"}


def oracle_dsdo(ch, mat, edges):
    parts = sorted(glob.glob(str(ROOT / "output" / f"oracle_ee_{mat}_{ch}" / "part_*" /
                                 f"inclusive_ee_{mat}_{ch}_s*.hepmc")))
    if not parts:
        raise SystemExit(f"no oracle hepmc for ee_{mat}_{ch}")
    nb = len(parts); bw = np.diff(edges); lo, hi = THETA_ACC
    sig = np.zeros(len(edges) - 1); var = np.zeros(len(edges) - 1)
    for f in parts:
        wtnb = hepmc_norm(f)["weight_to_nb"]
        om, wt = [], []
        for evt in parse_events(f):
            e = None
            for pid, st, p4 in evt["parts"]:
                if pid == 11 and st == 1 and (e is None or p4[0] > e[0]):
                    e = p4
            if e is None:
                continue
            ke = np.sqrt(e[1] ** 2 + e[2] ** 2 + e[3] ** 2)
            th = np.degrees(np.arccos(np.clip(e[3] / max(ke, 1e-9), -1, 1)))
            if lo <= th <= hi:
                om.append(E_BEAM - e[0]); wt.append(evt["w"] * wtnb)
        om = np.asarray(om); wt = np.asarray(wt) / nb
        idx = np.digitize(om, edges) - 1; m = (idx >= 0) & (idx < len(bw))
        np.add.at(sig, idx[m], wt[m]); np.add.at(var, idx[m], wt[m] ** 2)
    return sig / bw, np.sqrt(var) / bw


def adonis_dsdo(ch, mat, edges):
    parts = sorted(glob.glob(str(ROOT / "output" / f"ee_{mat}_{ch}_shards" / "part_*.npz")))
    if not parts:
        raise SystemExit(f"no ADoNIS shards for ee_{mat}_{ch}")
    ns = len(parts); bw = np.diff(edges); lo, hi = THETA_ACC
    sig = np.zeros(len(bw)); var = np.zeros(len(bw))
    for f in parts:
        z = np.load(f); m = (z["theta"] >= lo) & (z["theta"] <= hi)
        c = z["c"][m] / ns; idx = np.digitize(z["omega"][m], edges) - 1
        sel = (idx >= 0) & (idx < len(bw))
        np.add.at(sig, idx[sel], c[sel]); np.add.at(var, idx[sel], c[sel] ** 2)
    return sig / bw, np.sqrt(var) / bw


def main():
    ch = sys.argv[1] if len(sys.argv) > 1 else "qe"
    mat = sys.argv[2] if len(sys.argv) > 2 else "C"
    hi_om = 500 if ch == "qe" else 800
    edges = np.linspace(0, hi_om, (26 if ch == "qe" else 33))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    da, ea = oracle_dsdo(ch, mat, edges)
    dd, ed = adonis_dsdo(ch, mat, edges)
    dw = np.diff(edges)
    print(f"ee {ch} {mat}: oracle sigma_acc={(da*dw).sum():.1f} nb  ADoNIS={(dd*dw).sum():.1f} nb")
    m = (da > 0) & (dd > 0)
    chi2 = float(np.sum((dd[m] - da[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
    print(f"  dsigma/domega chi2/ndf = {chi2/max(ndf,1):.2f}  (ndf={ndf})   integral ratio = "
          f"{(dd*dw).sum()/max((da*dw).sum(),1e-9):.3f}")
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 6.2), height_ratios=[3, 1], sharex=True)
    col = "tab:red" if ch == "qe" else "tab:blue"
    a0.errorbar(ctr, da, ea, fmt="s", ms=4, color="k", label="ACHILLES oracle")
    a0.errorbar(ctr, dd, ed, fmt="o", ms=3, color=col, label="ADoNIS")
    a0.set_ylabel(r"$d\sigma/d\omega$ [nb/MeV]"); a0.legend(fontsize=9)
    a0.set_title(f"(e,e') {ch.upper()} on ${NUC[mat]}$, E=2.222 GeV, "
                 f"$\\theta_e\\in[14,17]^\\circ$   $\\chi^2$/ndf={chi2/max(ndf,1):.2f}", fontsize=10)
    r = np.where(da > 0, dd / np.where(da == 0, np.nan, da), np.nan)
    a1.errorbar(ctr, r, np.where(da > 0, ed / da, np.nan), fmt="o", ms=3, color=col)
    a1.axhline(1, color="k", lw=0.8); a1.set_ylim(0.5, 1.5)
    a1.set_ylabel("ADO/ACH"); a1.set_xlabel(r"$\omega = E - E'$ [MeV]")
    fig.tight_layout()
    out = ROOT / "output" / "paper" / f"xval_ee_{mat}_{ch}.png"
    fig.savefig(out, dpi=150); print(f"[fig] {out}")


if __name__ == "__main__":
    main()
