"""Validation gate for the ADoNIS inclusive (e,e') QE driver (adonis/xsec/ee_xsec.py) against the
ACHILLES oracle (output/oracle_ee_C_{qe}, run_inclusive_ee_C_qe.yml).

Both sides -> dsigma/domega [nb/MeV] inside the theta_e' HardCut acceptance, with statistical errors.
The comparison is chi2/ndf folding BOTH sides' errors (ACHILLES oracle MC + ADoNIS MC), matching the
convention used for the neutrino + hadron-beam validations.  Writes output/paper/xval_ee_C_qe.png.

Usage:  python -m analysis.beams.ee_validate [n_per_species]  (default 4_000_000)
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.utils.hepmc import parse_events, hepmc_norm   # noqa: E402
from adonis.xsec import ee_xsec as EE                         # noqa: E402

ORACLE = ROOT / "output" / "oracle_ee_C_qe"
OUT = ROOT / "output" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
E_BEAM = EE.E_BEAM_JLAB
LO, HI = EE.THETA_ACC


def oracle_dsdo(edges, theta_acc=EE.THETA_ACC):
    """ACHILLES dsigma/domega [nb/MeV] + stat error, summed over all seed parts.  Each part is an
    INDEPENDENT sigma estimate (its own GenCrossSection), so parts are averaged (weights /n_parts)."""
    parts = sorted(glob.glob(str(ORACLE / "part_*" / "inclusive_ee_C_qe_s*.hepmc")))
    if not parts:
        raise SystemExit(f"no oracle hepmc under {ORACLE}")
    nb = len(parts)
    bw = np.diff(edges)
    sig = np.zeros(len(edges) - 1); var = np.zeros(len(edges) - 1)
    lo, hi = theta_acc
    for f in parts:
        wtnb = hepmc_norm(f)["weight_to_nb"]
        om, wt = [], []
        for evt in parse_events(f):
            e_out = None
            for pid, status, p4 in evt["parts"]:
                if pid == 11 and status == 1:                    # scattered electron (final)
                    if e_out is None or p4[0] > e_out[0]:
                        e_out = p4
            if e_out is None:
                continue
            ke = np.sqrt(e_out[1] ** 2 + e_out[2] ** 2 + e_out[3] ** 2)
            th = np.degrees(np.arccos(np.clip(e_out[3] / max(ke, 1e-9), -1, 1)))
            if not (lo <= th <= hi):
                continue
            om.append(E_BEAM - e_out[0]); wt.append(evt["w"] * wtnb)
        om = np.asarray(om); wt = np.asarray(wt) / nb            # /n_parts: average of independent estimates
        idx = np.digitize(om, edges) - 1
        m = (idx >= 0) & (idx < len(bw))
        np.add.at(sig, idx[m], wt[m])
        np.add.at(var, idx[m], wt[m] ** 2)                       # sum w^2 -> Poisson stat variance
    return sig / bw, np.sqrt(var) / bw


def adonis_dsdo(res, edges, theta_acc=EE.THETA_ACC):
    """ADoNIS dsigma/domega [nb/MeV] + MC stat error (sqrt of sum c^2 per bin)."""
    lo, hi = theta_acc
    m = (res["theta"] >= lo) & (res["theta"] <= hi)
    om = res["omega"][m]; c = res["c"][m]
    bw = np.diff(edges)
    idx = np.digitize(om, edges) - 1
    sel = (idx >= 0) & (idx < len(bw))
    sig = np.zeros(len(bw)); var = np.zeros(len(bw))
    np.add.at(sig, idx[sel], c[sel])
    np.add.at(var, idx[sel], c[sel] ** 2)
    return sig / bw, np.sqrt(var) / bw


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    edges = np.linspace(0, 500, 26)                              # 20 MeV bins over the QE region
    ctr = 0.5 * (edges[:-1] + edges[1:])
    da, ea = oracle_dsdo(edges)
    print(f"ACHILLES oracle: sigma_acc = {(da*np.diff(edges)).sum():.2f} nb")
    res = EE.generate(n, material="C")
    dd, ed = adonis_dsdo(res, edges)
    print(f"ADoNIS driver:   sigma_acc = {(dd*np.diff(edges)).sum():.2f} nb  (n={n}/species)")
    m = (da > 0) & (dd > 0)
    chi2 = float(np.sum((dd[m] - da[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
    print(f"\ndsigma/domega chi2/ndf = {chi2/max(ndf,1):.2f}  (chi2={chi2:.1f}, ndf={ndf}, both-sides errors)")
    # figure
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 6.2), height_ratios=[3, 1], sharex=True)
    a0.errorbar(ctr, da, ea, fmt="s", ms=4, label="ACHILLES oracle", color="k")
    a0.errorbar(ctr, dd, ed, fmt="o", ms=3, label="ADoNIS", color="tab:red")
    a0.set_ylabel(r"$d\sigma/d\omega$ [nb/MeV]"); a0.legend(fontsize=9)
    a0.set_title(f"inclusive (e,e') QE on $^{{12}}$C, E=2.222 GeV, "
                 f"$\\theta_e\\in[{LO:.0f},{HI:.0f}]^\\circ$   "
                 f"$\\chi^2$/ndf={chi2/max(ndf,1):.2f}", fontsize=10)
    r = np.where(da > 0, dd / np.where(da == 0, np.nan, da), np.nan)
    rerr = np.where(da > 0, np.sqrt((ed / da) ** 2 + (dd * ea / da ** 2) ** 2), np.nan)
    a1.errorbar(ctr, r, rerr, fmt="o", ms=3, color="tab:red")
    a1.axhline(1, color="k", lw=0.8); a1.set_ylim(0.8, 1.2)
    a1.set_ylabel("ADO/ACH"); a1.set_xlabel(r"$\omega = E - E'$ [MeV]")
    fig.tight_layout()
    p = OUT / "xval_ee_C_qe.png"; fig.savefig(p, dpi=150); print(f"[fig] {p}")


if __name__ == "__main__":
    main()
