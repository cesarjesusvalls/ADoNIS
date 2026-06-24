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

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/multiplicity_xsec_C.png"
MU_WIN = [250.0, 7000.0]; COS70 = float(np.cos(np.deg2rad(70.0)))      # matrix muon acceptance
ADO_QE = "data/oracle/t2k_cc0pi_engine_rich_cv5_batch02.npz"
ADO_RES = "data/oracle/t2k_cc1pi_engine_rich_cv5_batch02.npz"
ACH = "data/oracle/t2k_cc1pi_rich_ach_FSI_proc.npz"
# (label, ado bank field, achilles -> count)
SPECIES = ["N(p)", "N(n)", "N(pi+)", "N(pi0)", "N(pi-)"]
ADO_FIELD = {"N(p)": "n_p", "N(n)": "n_n", "N(pi+)": "n_pip", "N(pi0)": "n_pi0", "N(pi-)": "n_pim"}


def mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > MU_WIN[0]) & (p < MU_WIN[1]) & (cz > COS70)


def ado(path):
    b = dict(np.load(path, allow_pickle=True))
    m = mu_mask(b["mu"]) & (b["w"] > 0)
    return {lab: b[ADO_FIELD[lab]][m] for lab in SPECIES}, b["w"][m]


def ach(procs):
    d = dict(np.load(ACH, allow_pickle=True))
    sel = np.isin(d["proc"], procs) & mu_mask(d["mu"])
    w = d["w"][sel] * float(d["weight_to_nb"])
    pp4 = d["prot_p4"][sel]; pip = d["pi_pid"][sel]
    n_p = (np.linalg.norm(pp4[:, :, 1:], axis=2) > 1e-6).sum(1)
    return {"N(p)": n_p, "N(pi+)": (pip == 211).sum(1), "N(pi0)": (pip == 111).sum(1),
            "N(pi-)": (pip == -211).sum(1)}, w


def dsig(counts, w, nmax=6):
    """absolute dsigma per multiplicity bin (nb) + weighted statistical error sqrt(sum w^2)."""
    c = np.clip(counts, 0, nmax)
    val = np.array([w[c == k].sum() for k in range(nmax + 1)])
    err = np.array([np.sqrt((w[c == k] ** 2).sum()) for k in range(nmax + 1)])
    return val, err


def main():
    print("loading banks ...", flush=True)
    aQ, wQ = ado(ADO_QE); aR, wR = ado(ADO_RES)
    hQ, hwQ = ach([200]); hR, hwR = ach([401, 402])
    nmax = 6; edges = np.arange(nmax + 2.0); ctr = 0.5 * (edges[1:] + edges[:-1])   # bin k = [k,k+1)
    # top dsigma/dN + bottom ACH/ADO ratio per species, exactly like the cross-section matrix panels
    fig, axes = plt.subplots(4, 5, figsize=(23, 11), height_ratios=[3, 1, 3, 1])
    for (chan, A, wa, H, wh, r0, r1) in [("QE", aQ, wQ, hQ, hwQ, 0, 1), ("RES", aR, wR, hR, hwR, 2, 3)]:
        for c, lab in enumerate(SPECIES):
            a0, a1 = axes[r0, c], axes[r1, c]
            if lab in H:
                chi2_ratio_panel(a0, a1, edges, {"values": H[lab], "w": wh}, {"values": A[lab], "w": wa},
                                 label=f"{chan}: {lab}", ado_label="ADoNIS", ref_label="ACHILLES")
            else:                                            # neutrons: ADoNIS only (ACHILLES bank has none)
                dd, ed = hist_with_errors(A[lab], wa, edges)
                a0.errorbar(ctr, dd, yerr=ed, fmt="s", color="C0", ms=3, capsize=2, lw=0.9, label="ADoNIS")
                a0.set_title(f"{chan}: {lab}", fontsize=9); a0.set_ylim(bottom=0)
                a1.axhline(1.0, ls="--", color="green", lw=0.7); a1.set_ylim(0.5, 1.6)
                a1.text(0.5, 0.4, "no ACHILLES n", transform=a1.transAxes, ha="center", fontsize=8, color="r")
                a1.set_xlabel(f"{chan}: {lab}", fontsize=8)
    axes[0, 0].set_ylabel("QE  dσ/dN [nb]"); axes[2, 0].set_ylabel("RES  dσ/dN [nb]")
    axes[1, 0].set_ylabel("ACH/ADO"); axes[3, 0].set_ylabel("ACH/ADO"); axes[0, 0].legend(fontsize=8)
    fig.suptitle("dσ vs final-state multiplicity (C, muon-acceptance CC-inclusive): ADoNIS vs ACHILLES (+ ACH/ADO ratio)")
    fig.tight_layout(); fig.savefig(OUT, dpi=110); print(f"wrote {OUT}", flush=True)
    for chan, A, wa, H, wh in [("QE", aQ, wQ, hQ, hwQ), ("RES", aR, wR, hR, hwR)]:
        print(f"--- {chan}: total sigma ADoNIS={wa.sum():.4e}  ACHILLES={wh.sum():.4e}  ACH/ADO={wh.sum()/wa.sum():.3f}", flush=True)
        for lab in SPECIES:
            a, ae = dsig(A[lab], wa, nmax); hh = dsig(H[lab], wh, nmax)[0] if lab in H else None
            print(f"  {lab:7s} ado N0..3: {a[:4]}" + (f"   ach: {hh[:4]}" if hh is not None else "  (ado-only)"), flush=True)


if __name__ == "__main__":
    main()
