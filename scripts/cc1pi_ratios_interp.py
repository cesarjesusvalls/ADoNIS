"""DEFINITIVE bilinear-vs-spline cc1pi_ratios: forces DC.BATCH_INTERP at RUNTIME for each interp
(no file-default ambiguity), runs the SAME accumulate (RES-C 120k x4 + RES-H 40k x4) for both,
prints the 7-variable ACH/ADO chi2/ndf + integral table for each, and a per-variable delta.
Same fixed seeds -> the ONLY difference between the two tables is the amplitude interpolation.

Usage: python scripts/cc1pi_ratios_interp.py [NRES=120000] [NSEED=4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import adonis.xsec.dcc_current as DC
import scripts.cc1pi_fig_tki as F
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NH = 40000
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")
VARS = [
    ("pn",      np.array([0, 120, 240, 600, 1500.0])),
    ("dptt",    np.array([-700, -300, -100, 100, 300, 700.0])),
    ("dalphat", np.linspace(0, np.pi, 7)),
    ("W",       np.linspace(1080, 1700, 13)),
    ("Q2",      np.linspace(0, 1.5, 13)),
    ("pi_p",    np.linspace(150, 1200, 13)),
    ("lp_p",    np.linspace(450, 1200, 13)),
]


def accumulate():
    cells = {}
    for cell, fn, n in (("RES-C", F.res_C, NRES), ("RES-H", F.res_H, NH)):
        parts = [fn(n, sd) for sd in range(NSEED)]
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        d["w"] = d["w"] / NSEED
        cells[cell] = d
    return {k: np.concatenate([cells[c][k] for c in cells]) for k in KEYS + ("w",)}


def table(ado, ach, ach_w):
    rows = {}
    for key, edges in VARS:
        bw = np.diff(edges)
        av = np.asarray(ach[key]); dv = ado[key]
        if key == "Q2":
            av = av / 1e6
        da, _ = np.histogram(av, bins=edges, weights=ach_w)
        ea2, _ = np.histogram(av, bins=edges, weights=ach_w ** 2)
        dd, _ = np.histogram(dv, bins=edges, weights=ado["w"])
        ed2, _ = np.histogram(dv, bins=edges, weights=ado["w"] ** 2)
        da, ea, dd, ed = da / bw, np.sqrt(ea2) / bw, dd / bw, np.sqrt(ed2) / bw
        m = (da > 0) & (dd > 0)
        chi2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
        iA = float(np.sum(da * bw)); iD = float(np.sum(dd * bw))
        rows[key] = (chi2 / max(ndf, 1), iA / iD, iD)
    return rows


def main():
    ach = np.load("data/oracle/t2k_cc1pi_tki_achilles.npz")
    ach_w = np.asarray(ach["w"]) * weight_to_nb_of(ach)
    res = {}
    for interp in ("bilinear", "spline"):
        DC.BATCH_INTERP = interp
        print(f"\n===== BATCH_INTERP = {interp} (runtime-forced) =====", flush=True)
        ado = accumulate()
        print(f"  ADoNIS selected sigma_CC1pi = {ado['w'].sum():.5e} nb", flush=True)
        res[interp] = table(ado, ach, ach_w)
        for key, _ in VARS:
            c2, ir, iD = res[interp][key]
            print(f"    {key:10s} chi2/ndf={c2:7.2f}  ACH/ADO={ir:.3f}", flush=True)
    print(f"\n{'variable':10s}  {'chi2 bilin':>10} {'chi2 spline':>11}  {'ACH/ADO bilin':>13} {'spline':>8}")
    for key, _ in VARS:
        b, s = res["bilinear"][key], res["spline"][key]
        print(f"{key:10s}  {b[0]:10.2f} {s[0]:11.2f}  {b[1]:13.3f} {s[1]:8.3f}")


if __name__ == "__main__":
    main()
