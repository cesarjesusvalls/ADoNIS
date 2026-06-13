"""ABLATION: vary the proton-candidate set (KNOCKOUT_MODE) and measure the impact on the CC1pi
high-W/pi_p tail.  Modes (cc1pi_fig_tki.KNOCKOUT_MODE):
  full         = {RES nucleon, leading pion-scatter knockout, secondary knockout}  (faithful)
  no_secondary = drop the secondary knockout (tests the bounded-recursion approximation)
  res_only     = RES nucleon only (removes the n->n pi+-via-knockout path entirely)
Same seeds across modes -> the cascade walk is identical; only the proton candidate set differs.
Diagnostic (measures the knockout/recursion CONTRIBUTION to the tail), not a faithfulness change.

Usage: python scripts/cc1pi_knockout_ablation.py [NRES=80000] [NSEED=4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scripts.cc1pi_fig_tki as F
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NH = 40000
KEYS = ("pn", "dptt", "dalphat", "W", "Q2", "pi_p", "lp_p")
VARS = [("pn", np.array([0, 120, 240, 600, 1500.0])),
        ("dptt", np.array([-700, -300, -100, 100, 300, 700.0])),
        ("dalphat", np.linspace(0, np.pi, 7)),
        ("W", np.linspace(1080, 1700, 13)),
        ("Q2", np.linspace(0, 1.5, 13)),
        ("pi_p", np.linspace(150, 1200, 13)),
        ("lp_p", np.linspace(450, 1200, 13))]


def accumulate():
    cells = {}
    for cell, fn, n in (("RES-C", F.res_C, NRES), ("RES-H", F.res_H, NH)):
        parts = [fn(n, sd) for sd in range(NSEED)]
        d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("w",)}
        d["w"] = d["w"] / NSEED
        cells[cell] = d
    return {k: np.concatenate([cells[c][k] for c in cells]) for k in KEYS + ("w",)}


def table(ado, ach, aw):
    out = {}
    for key, edges in VARS:
        bw = np.diff(edges); ctr = 0.5 * (edges[1:] + edges[:-1])
        av = np.asarray(ach[key]) / (1e6 if key == "Q2" else 1.0)
        da, _ = np.histogram(av, edges, weights=aw); ea2, _ = np.histogram(av, edges, weights=aw ** 2)
        dd, _ = np.histogram(ado[key], edges, weights=ado["w"]); ed2, _ = np.histogram(ado[key], edges, weights=ado["w"] ** 2)
        da, ea, dd, ed = da / bw, np.sqrt(ea2) / bw, dd / bw, np.sqrt(ed2) / bw
        m = (da > 0) & (dd > 0)
        c2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))) / max(int(m.sum()), 1)
        hi = ctr > 1390 if key == "W" else (ctr > 600 if key == "pi_p" else np.ones_like(ctr, bool))
        iA = float(np.sum((da * bw))); iD = float(np.sum((dd * bw)))
        iAh = float(np.sum((da * bw)[hi])); iDh = float(np.sum((dd * bw)[hi]))
        out[key] = (c2, iA / max(iD, 1e-30), iAh / max(iDh, 1e-30))
    return out


def main():
    ach = np.load("data/oracle/t2k_cc1pi_tki_achilles.npz")
    aw = np.asarray(ach["w"]) * weight_to_nb_of(ach)
    res = {}
    for mode in ("full", "no_secondary", "res_only"):
        F.KNOCKOUT_MODE = mode
        print(f"\n===== KNOCKOUT_MODE = {mode} =====", flush=True)
        ado = accumulate()
        res[mode] = (table(ado, ach, aw), float(ado["w"].sum()))
        print(f"  selected sigma = {res[mode][1]:.4e} nb", flush=True)
        for key, _ in VARS:
            c2, ir, irh = res[mode][0][key]
            print(f"    {key:9s} chi2/ndf={c2:7.2f}  ACH/ADO={ir:.3f}", flush=True)
    print(f"\n{'':9s} | " + " | ".join(f"{m:>13}" for m in res))
    print(f"{'sigma':9s} | " + " | ".join(f"{res[m][1]:13.4e}" for m in res))
    for key, _ in VARS:
        print(f"{key+' chi2':9s} | " + " | ".join(f"{res[m][0][key][0]:13.2f}" for m in res))
    print("--- high-W(>1390)/high-pi_p(>600) ACH/ADO ---")
    for key in ("W", "pi_p"):
        print(f"{key+' hi':9s} | " + " | ".join(f"{res[m][0][key][2]:13.3f}" for m in res))


if __name__ == "__main__":
    main()
