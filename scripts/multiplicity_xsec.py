"""Absolute differential cross section vs final-state multiplicity, QE-only and RES-only, ADoNIS vs
ACHILLES -- the SAME absolute comparison the cc0pi/cc1pi matrix does (NO normalization), extended to a
new observable (particle multiplicity).  Selection = muon-acceptance CC-inclusive (mu window + forward
cos, the matrix's muon cut), NO hadronic/topology cut.  Absolute weights match the matrix: ADoNIS = bank
w (already nb-scaled by gen_events); ACHILLES = bank w * weight_to_nb.

ADoNIS multiplicity from the bank fields n_p/n_n/n_pip/n_pi0/n_pim (full escaped final state, M_out=24).
ACHILLES from prot_p4 (n_p) + pi_pid (pions); ACHILLES bank has NO neutrons -> N(n) is ADoNIS-only.

Run: python -u scripts/multiplicity_xsec.py [out.png]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.plotting import chi2_ratio_panel, hist_with_errors

import glob
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/multiplicity_xsec_C.png"
MODE = os.environ.get("MODE", "fsi").lower()                           # fsi | nofsi
MU_WIN = [250.0, 7000.0]; COS70 = float(np.cos(np.deg2rad(70.0)))      # matrix muon acceptance
_suf = "_nofsi" if MODE == "nofsi" else ""
ADO_DIR = os.environ.get("ADO_DIR", "output/adonis")                   # ADoNIS bank dir override
MAT = os.environ.get("MAT", "C")                                       # material in the bank name: C | Ar
FLUX = os.environ.get("FLUX", "t2k")                                   # flux in the bank name (t2k only today)
TAG = os.environ.get("TAG", "")                                        # extra variant tag (canonical = empty)
ADO_QE = sorted(glob.glob(f"{ADO_DIR}/{FLUX}_{MAT}_cc0pi{TAG}{_suf}_batch*.npz"))   # merge ALL QE batches
ADO_RES = sorted(glob.glob(f"{ADO_DIR}/{FLUX}_{MAT}_cc1pi{TAG}{_suf}_batch*.npz"))
ACH = os.environ.get("ACH_BANK",                                       # ACHILLES proc bank override
                     f"data/oracle/t2k_cc1pi_rich_ach_{'nofsi' if MODE == 'nofsi' else 'FSI'}_proc.npz")
# (label, ado bank field, achilles -> count)
SPECIES = ["N(p)", "N(n)", "N(pi+)", "N(pi0)", "N(pi-)"]
ADO_FIELD = {"N(p)": "n_p", "N(n)": "n_n", "N(pi+)": "n_pip", "N(pi0)": "n_pi0", "N(pi-)": "n_pim"}
CHANNEL = os.environ.get("CHANNEL", "qe").lower()                      # qe | res
ADO = ADO_QE if CHANNEL == "qe" else ADO_RES
PROC = [200] if CHANNEL == "qe" else [401, 402]                        # ACHILLES proc: 200=QE, 401/402=RES
CLABEL = CHANNEL.upper()


def mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > MU_WIN[0]) & (p < MU_WIN[1]) & (cz > COS70)


def ado(paths):
    """Merge >=1 homogeneous batches: concatenate events, divide w by n_batches so sum(w) stays sigma
    (each batch's w already sums to sigma; the merge is an average of independent dsigma estimates)."""
    if isinstance(paths, str):
        paths = [paths]
    paths = [p for p in paths if "n_p" in np.load(p, allow_pickle=True).files]   # skip pre-mult banks
    print(f"  merging {len(paths)} batch(es): {[os.path.basename(p) for p in paths]}", flush=True)
    nb = len(paths); cnt = {lab: [] for lab in SPECIES}; ws = []
    for p in paths:
        b = dict(np.load(p, allow_pickle=True))
        m = mu_mask(b["mu"]) & (b["w"] > 0)
        for lab in SPECIES:
            cnt[lab].append(b[ADO_FIELD[lab]][m])
        ws.append(b["w"][m])
    return {lab: np.concatenate(cnt[lab]) for lab in SPECIES}, np.concatenate(ws) / nb


def ach(procs):
    d = dict(np.load(ACH, allow_pickle=True))
    sel = np.isin(d["proc"], procs) & mu_mask(d["mu"])
    w = d["w"][sel] * float(d["weight_to_nb"])
    pp4 = d["prot_p4"][sel]; pip = d["pi_pid"][sel]
    n_p = (np.linalg.norm(pp4[:, :, 1:], axis=2) > 1e-6).sum(1)
    out = {"N(p)": n_p, "N(pi+)": (pip == 211).sum(1), "N(pi0)": (pip == 111).sum(1),
           "N(pi-)": (pip == -211).sum(1)}
    if "neut_p4" in d:                                          # neutrons now stored (extract_cc1pi_rich)
        np4 = d["neut_p4"][sel]
        out["N(n)"] = (np.linalg.norm(np4[:, :, 1:], axis=2) > 1e-6).sum(1)
    return out, w


def dsig(counts, w, nmax=6):
    """absolute dsigma per multiplicity bin (nb) + weighted statistical error sqrt(sum w^2)."""
    c = np.clip(counts, 0, nmax)
    val = np.array([w[c == k].sum() for k in range(nmax + 1)])
    err = np.array([np.sqrt((w[c == k] ** 2).sum()) for k in range(nmax + 1)])
    return val, err


def main():
    print(f"loading banks (channel={CLABEL}) ...", flush=True)
    aQ, wQ = ado(ADO)
    hQ, hwQ = ach(PROC)
    nmax = 6; edges = np.arange(nmax + 2.0); ctr = 0.5 * (edges[1:] + edges[:-1])   # bin k = [k,k+1)
    # top dsigma/dN + bottom ACH/ADO ratio per species, exactly like the cross-section matrix panels
    fig, axes = plt.subplots(2, 5, figsize=(23, 6), height_ratios=[3, 1], sharex="col")
    for c, lab in enumerate(SPECIES):
        a0, a1 = axes[0, c], axes[1, c]
        a0.set_xlim(edges[0], edges[-1])                        # top + ratio share this range (sharex)
        if lab in hQ:
            chi2_ratio_panel(a0, a1, edges, {"values": hQ[lab], "w": hwQ}, {"values": aQ[lab], "w": wQ},
                             label=f"{CLABEL}: {lab}", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
        else:                                                  # (only if ACHILLES bank lacks this species)
            dd, ed = hist_with_errors(aQ[lab], wQ, edges)
            a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADoNIS")
            a0.set_title(f"{CLABEL}: {lab}", fontsize=9); a0.set_ylim(bottom=0)
            a1.axhline(1.0, ls="--", color="green", lw=0.7); a1.set_ylim(0.5, 1.6)
            a1.text(0.5, 0.4, "no ACHILLES", transform=a1.transAxes, ha="center", fontsize=8, color="r")
            a1.set_xlabel(f"{CLABEL}: {lab}", fontsize=8)
    axes[0, 0].set_ylabel(f"{CLABEL}  dσ/dN [nb]"); axes[1, 0].set_ylabel("ACH/ADO"); axes[0, 0].legend(fontsize=8)
    nb = len([p for p in ADO if "n_p" in np.load(p, allow_pickle=True).files])
    fig.suptitle(f"{CLABEL} dσ vs final-state multiplicity (C, muon-acceptance CC-inclusive; {nb} batches; "
                 f"{'NO FSI' if MODE == 'nofsi' else 'FSI'}): ADoNIS vs ACHILLES")
    fig.tight_layout(); fig.savefig(OUT, dpi=110); print(f"wrote {OUT}", flush=True)
    print(f"--- {CLABEL}: total sigma ADoNIS={wQ.sum():.4e}  ACHILLES={hwQ.sum():.4e}  ACH/ADO={hwQ.sum()/wQ.sum():.3f}", flush=True)
    for lab in SPECIES:
        a, ae = dsig(aQ[lab], wQ, nmax); hh = dsig(hQ[lab], hwQ, nmax)[0] if lab in hQ else None
        print(f"  {lab:7s} ado N0..3: {a[:4]}" + (f"   ach: {hh[:4]}" if hh is not None else "  (ado-only)"), flush=True)


if __name__ == "__main__":
    main()
