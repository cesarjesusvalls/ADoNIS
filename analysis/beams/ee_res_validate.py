"""Validation gate for the ADoNIS inclusive (e,e') RES driver (adonis/xsec/res_ee_xsec.py) against the
ACHILLES oracle output/oracle_ee_C_res (run_inclusive_ee_C_res.yml).  dsigma/domega [nb/MeV] both sides,
inside the theta_e' HardCut, chi2/ndf folding both errors.  Writes output/paper/xval_ee_C_res.png.

Usage:  python -m analysis.beams.ee_res_validate [n_per_channel]
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
from analysis.utils.hepmc import parse_events, hepmc_norm      # noqa: E402
from adonis.xsec import res_ee_xsec as RE                       # noqa: E402

ORACLE = ROOT / "output" / "oracle_ee_C_res"
OUT = ROOT / "output" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
E_BEAM = RE.E_BEAM_JLAB
LO, HI = RE.THETA_ACC


def oracle_dsdo(edges, theta_acc=RE.THETA_ACC):
    parts = sorted(glob.glob(str(ORACLE / "part_*" / "inclusive_ee_C_res_s*.hepmc")))
    if not parts:
        raise SystemExit(f"no oracle hepmc under {ORACLE}")
    nb = len(parts); bw = np.diff(edges)
    sig = np.zeros(len(edges) - 1); var = np.zeros(len(edges) - 1)
    lo, hi = theta_acc
    for f in parts:
        wtnb = hepmc_norm(f)["weight_to_nb"]
        om, wt = [], []
        for evt in parse_events(f):
            e_out = None
            for pid, status, p4 in evt["parts"]:
                if pid == 11 and status == 1 and (e_out is None or p4[0] > e_out[0]):
                    e_out = p4
            if e_out is None:
                continue
            ke = np.sqrt(e_out[1] ** 2 + e_out[2] ** 2 + e_out[3] ** 2)
            th = np.degrees(np.arccos(np.clip(e_out[3] / max(ke, 1e-9), -1, 1)))
            if not (lo <= th <= hi):
                continue
            om.append(E_BEAM - e_out[0]); wt.append(evt["w"] * wtnb)
        om = np.asarray(om); wt = np.asarray(wt) / nb
        idx = np.digitize(om, edges) - 1
        m = (idx >= 0) & (idx < len(bw))
        np.add.at(sig, idx[m], wt[m]); np.add.at(var, idx[m], wt[m] ** 2)
    return sig / bw, np.sqrt(var) / bw


def adonis_dsdo(res, edges, theta_acc=RE.THETA_ACC):
    lo, hi = theta_acc
    m = (res["theta"] >= lo) & (res["theta"] <= hi)
    idx = np.digitize(res["omega"][m], edges) - 1
    bw = np.diff(edges); sel = (idx >= 0) & (idx < len(bw))
    sig = np.zeros(len(bw)); var = np.zeros(len(bw))
    c = res["c"][m]
    np.add.at(sig, idx[sel], c[sel]); np.add.at(var, idx[sel], c[sel] ** 2)
    return sig / bw, np.sqrt(var) / bw


def _load_shards(shard_dir):
    parts = sorted(glob.glob(str(Path(shard_dir) / "part_*.npz")))
    if not parts:
        raise SystemExit(f"no shards in {shard_dir}")
    d = {k: [] for k in ("c", "omega", "theta")}
    for f in parts:
        z = np.load(f)
        for k in d:
            d[k].append(z[k])
    n_shards = len(parts)
    out = {k: np.concatenate(v) for k, v in d.items()}
    out["c"] = out["c"] / n_shards        # each shard is an independent sigma estimate -> average
    print(f"loaded {n_shards} shards, {len(out['c'])} events")
    return out


def main():
    edges = np.linspace(0, 800, 33)                             # 25 MeV bins over the RES/Delta region
    ctr = 0.5 * (edges[:-1] + edges[1:])
    da, ea = oracle_dsdo(edges)
    print(f"ACHILLES oracle RES: sigma_acc = {(da*np.diff(edges)).sum():.2f} nb")
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        shard_dir = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "output" / "res_ee_shards")
        res = _load_shards(shard_dir); n = "shards"
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
        res = RE.generate(n, material="C")
    dd, ed = adonis_dsdo(res, edges)
    print(f"ADoNIS RES driver:   sigma_acc = {(dd*np.diff(edges)).sum():.2f} nb  (n={n}/channel)")
    m = (da > 0) & (dd > 0)
    chi2 = float(np.sum((dd[m] - da[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
    print(f"\ndsigma/domega chi2/ndf = {chi2/max(ndf,1):.2f}  (chi2={chi2:.1f}, ndf={ndf}, both-sides)")
    print(f"ratio ADO/ACH (integral) = {(dd*np.diff(edges)).sum()/max((da*np.diff(edges)).sum(),1e-9):.3f}")
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 6.2), height_ratios=[3, 1], sharex=True)
    a0.errorbar(ctr, da, ea, fmt="s", ms=4, label="ACHILLES oracle", color="k")
    a0.errorbar(ctr, dd, ed, fmt="o", ms=3, label="ADoNIS", color="tab:blue")
    a0.set_ylabel(r"$d\sigma/d\omega$ [nb/MeV]"); a0.legend(fontsize=9)
    a0.set_title(f"inclusive (e,e') RES on $^{{12}}$C, E=2.222 GeV, "
                 f"$\\theta_e\\in[{LO:.0f},{HI:.0f}]^\\circ$   $\\chi^2$/ndf={chi2/max(ndf,1):.2f}", fontsize=10)
    r = np.where(da > 0, dd / np.where(da == 0, np.nan, da), np.nan)
    a1.errorbar(ctr, r, np.where(da > 0, ed / da, np.nan), fmt="o", ms=3, color="tab:blue")
    a1.axhline(1, color="k", lw=0.8); a1.set_ylim(0.5, 1.5)
    a1.set_ylabel("ADO/ACH"); a1.set_xlabel(r"$\omega = E - E'$ [MeV]")
    fig.tight_layout()
    p = OUT / "xval_ee_C_res.png"; fig.savefig(p, dpi=150); print(f"[fig] {p}")


if __name__ == "__main__":
    main()
