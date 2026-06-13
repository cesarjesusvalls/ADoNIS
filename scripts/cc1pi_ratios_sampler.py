"""DEFINITIVE 3-body-sampler A/B on the actual cc1pi_ratios diagnostic: isotropic vs ACHILLES
t-channel ThreeBodyMapper, BOTH with the faithful spline amps2.  Same fixed seeds -> the only
difference is the pion-split proposal.  Prints both 7-variable ACH/ADO chi2/ndf + integral tables.

Usage: python scripts/cc1pi_ratios_sampler.py [NRES=120000] [NSEED=4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import adonis.xsec.dcc_current as DC
import adonis.xsec.res_xsec as R
import scripts.cc1pi_fig_tki as F          # pins spline at import
from adonis.data.oracle.normalization import weight_to_nb_of

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
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
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        d["w"] = d["w"] / NSEED
        cells[cell] = d
    return {k: np.concatenate([cells[c][k] for c in cells]) for k in KEYS + ("w",)}


def table(ado, ach, ach_w):
    rows = {}
    for key, edges in VARS:
        bw = np.diff(edges)
        av = np.asarray(ach[key]) / (1e6 if key == "Q2" else 1.0); dv = ado[key]
        da, _ = np.histogram(av, bins=edges, weights=ach_w); ea2, _ = np.histogram(av, bins=edges, weights=ach_w ** 2)
        dd, _ = np.histogram(dv, bins=edges, weights=ado["w"]); ed2, _ = np.histogram(dv, bins=edges, weights=ado["w"] ** 2)
        da, ea, dd, ed = da / bw, np.sqrt(ea2) / bw, dd / bw, np.sqrt(ed2) / bw
        m = (da > 0) & (dd > 0)
        chi2 = float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))); ndf = int(m.sum())
        rows[key] = (chi2 / max(ndf, 1), float(np.sum(da * bw)) / float(np.sum(dd * bw)))
    return rows


def main():
    DC.BATCH_INTERP = "spline"
    ach = np.load("data/oracle/t2k_cc1pi_tki_achilles.npz")
    ach_w = np.asarray(ach["w"]) * weight_to_nb_of(ach)
    res = {}
    for samp in ("isotropic", "tchannel"):
        R.SAMPLER_3BODY = samp
        print(f"\n===== SAMPLER_3BODY = {samp} (spline amps2) =====", flush=True)
        ado = accumulate()
        print(f"  selected sigma_CC1pi = {ado['w'].sum():.5e} nb", flush=True)
        res[samp] = table(ado, ach, ach_w)
        for key, _ in VARS:
            print(f"    {key:10s} chi2/ndf={res[samp][key][0]:7.2f}  ACH/ADO={res[samp][key][1]:.3f}", flush=True)
    print(f"\n{'variable':10s}  {'chi2 iso':>9} {'chi2 tchan':>10}  {'ACH/ADO iso':>11} {'tchan':>7}")
    for key, _ in VARS:
        i, t = res["isotropic"][key], res["tchannel"][key]
        print(f"{key:10s}  {i[0]:9.2f} {t[0]:10.2f}  {i[1]:11.3f} {t[1]:7.3f}")


if __name__ == "__main__":
    main()
