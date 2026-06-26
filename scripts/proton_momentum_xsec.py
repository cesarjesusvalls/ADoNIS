"""Absolute differential cross section vs LEADING-proton momentum, QE-only, ADoNIS vs ACHILLES -- the
SAME absolute comparison the cc0pi/cc1pi matrix does (NO normalization), as a continuous observable to
complement the integer multiplicity (scripts/multiplicity_xsec.py).  Selection = muon-acceptance
CC-inclusive (mu window + forward cos), require >=1 proton; the leading proton is the highest-|p| proton.
Absolute weights match the matrix: ADoNIS = bank w (already nb-scaled by gen_events, /n_batches on merge);
ACHILLES = bank w * weight_to_nb.

MODE=fsi|nofsi env selects the FSI batches + FSI ACHILLES bank, or the _nofsi family.
Run: MODE=fsi python -u scripts/proton_momentum_xsec.py [out.png]
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.plotting import chi2_ratio_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/proton_momentum_xsec_C.png"
MODE = os.environ.get("MODE", "fsi").lower()
SPECIES = os.environ.get("SPECIES", "proton").lower()      # proton | neutron
ADO_FIELD = "neut" if SPECIES == "neutron" else "prot"
ACH_FIELD = "neut_p4" if SPECIES == "neutron" else "prot_p4"
SPL = "neutron" if SPECIES == "neutron" else "proton"
MU_WIN = [250.0, 7000.0]; COS70 = float(np.cos(np.deg2rad(70.0)))
_suf = "_nofsi" if MODE == "nofsi" else ""
ADO_DIR = os.environ.get("ADO_DIR", "output/adonis")
MAT = os.environ.get("MAT", "C"); FLUX = os.environ.get("FLUX", "t2k"); TAG = os.environ.get("TAG", "")
ADO_QE = sorted(glob.glob(f"{ADO_DIR}/{FLUX}_{MAT}_cc0pi{TAG}{_suf}_batch*.npz"))
ACH = os.environ.get("ACH_BANK",
                     f"data/oracle/t2k_cc1pi_rich_ach_{'nofsi' if MODE == 'nofsi' else 'FSI'}_proc.npz")
EDGES = np.linspace(0.0, 2000.0, 41)                       # 50 MeV bins; log-y shows the high-p tail


def mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > MU_WIN[0]) & (p < MU_WIN[1]) & (cz > COS70)


def ado():
    paths = [p for p in ADO_QE if ADO_FIELD in np.load(p, allow_pickle=True).files]
    print(f"  merging {len(paths)} batch(es)", flush=True)
    lp, ws = [], []
    for p in paths:
        b = dict(np.load(p, allow_pickle=True))
        m = mu_mask(b["mu"]) & (b["w"] > 0)
        plead = np.linalg.norm(b[ADO_FIELD][m][:, 0, 1:], axis=1)       # leading-species |p|
        has = plead > 0
        lp.append(plead[has]); ws.append(b["w"][m][has])
    return np.concatenate(lp), np.concatenate(ws) / len(paths)


def ach():
    d = dict(np.load(ACH, allow_pickle=True))
    sel = np.isin(d["proc"], [200]) & mu_mask(d["mu"])
    plead = np.linalg.norm(d[ACH_FIELD][sel][:, 0, 1:], axis=1)
    has = plead > 0
    return plead[has], d["w"][sel][has] * float(d["weight_to_nb"])


def main():
    print("loading banks ...", flush=True)
    aP, wa = ado(); hP, wh = ach()
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)
    chi2_ratio_panel(a0, a1, EDGES, {"values": hP, "w": wh}, {"values": aP, "w": wa},
                     label=rf"QE: lead $p_{{{SPL[0]}}}$ [MeV]", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
    a0.set_ylabel("QE  dσ/dp [nb/MeV]"); a1.set_ylabel("ACH/ADO"); a0.legend(fontsize=8)
    nb = len([p for p in ADO_QE if ADO_FIELD in np.load(p, allow_pickle=True).files])
    fig.suptitle(f"QE dσ/dp(lead {SPL}) (C, muon-acceptance CC-inclusive; {nb} batches; "
                 f"{'NO FSI' if MODE == 'nofsi' else 'FSI'}): ADoNIS vs ACHILLES")
    fig.tight_layout(); fig.savefig(OUT, dpi=120); print(f"wrote {OUT}", flush=True)
    print(f"--- QE lead-{SPL[0]}: sigma(>=1) ADoNIS={wa.sum():.4e}  ACHILLES={wh.sum():.4e}  ACH/ADO={wh.sum()/wa.sum():.3f}", flush=True)
    for q in (50, 90, 99):
        print(f"  p{q}: ADoNIS={np.percentile(aP, q):.0f}  ACHILLES={np.percentile(hP, q):.0f} MeV", flush=True)


if __name__ == "__main__":
    main()
