"""Sweep CC1pi+Np signal definitions from the RICH banks (pure re-binning) and tabulate, per def:
integral ACH/ADO, W & pi_p chi2/ndf, and the per-channel ACH/ADO (p->p pi+ / n->n pi+).
No regeneration -- every row is a re-bin of the same three banks.
Usage: python scripts/cc1pi_sweep.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scripts.cc1pi_signal as S

ADO = dict(np.load("data/oracle/t2k_cc1pi_rich_adonis.npz"))
ACH = {True: dict(np.load("data/oracle/t2k_cc1pi_rich_ach_FSI.npz")),
       False: dict(np.load("data/oracle/t2k_cc1pi_rich_ach_nofsi.npz"))}

DEFS = [
    ("FSI carbon (fixed)",            dict(S.DEFAULT)),
    ("FSI carbon recoil-N (old bug)", dict(S.DEFAULT, count_recoil_neutron=True)),
    ("FSI carbon native-only",        dict(S.DEFAULT, proton_source="native")),
    ("noFSI carbon (fixed)",          dict(S.DEFAULT, fsi=False)),
    ("noFSI carbon recoil-N",         dict(S.DEFAULT, fsi=False, count_recoil_neutron=True)),
    ("FSI CH (fixed)",                dict(S.DEFAULT, target="CH")),
    ("FSI hydrogen",                  dict(S.DEFAULT, target="hydrogen")),
]


def chi2(A, H, key, edges):
    bw = np.diff(edges)
    da, _ = np.histogram(H[key], edges, weights=H["w"]); ea2, _ = np.histogram(H[key], edges, weights=H["w"] ** 2)
    dd, _ = np.histogram(A[key], edges, weights=A["w"]); ed2, _ = np.histogram(A[key], edges, weights=A["w"] ** 2)
    da, ea, dd, ed = da / bw, np.sqrt(ea2) / bw, dd / bw, np.sqrt(ed2) / bw
    m = (da > 0) & (dd > 0)
    return float(np.sum((da[m] - dd[m]) ** 2 / (ea[m] ** 2 + ed[m] ** 2))) / max(int(m.sum()), 1)


Wed = np.linspace(1080, 1700, 13); Ped = np.linspace(150, 1200, 13)
print(f"{'definition':32s} {'ACH/ADO':>8} {'W chi2':>7} {'pip chi2':>8} {'p->ppi+':>8} {'n->npi+':>8}")
for name, sd in DEFS:
    A = S.ado_select(ADO, sd)
    H = S.ach_select(ACH[sd["fsi"]], sd); H["w"] = H["w"] * float(ACH[sd["fsi"]]["weight_to_nb"])
    iA, iD = H["w"].sum(), A["w"].sum()
    wc, pc = chi2(A, H, "W", Wed), chi2(A, H, "pi_p", Ped)
    def chan(d, c): return d["w"][d["chan"] == c].sum()
    rpp = chan(H, 2212) / max(chan(A, 2212), 1e-30)
    rnn = chan(H, 2112) / max(chan(A, 2112), 1e-30)
    print(f"{name:32s} {iA/max(iD,1e-30):8.3f} {wc:7.1f} {pc:8.1f} {rpp:8.3f} {rnn:8.3f}")
