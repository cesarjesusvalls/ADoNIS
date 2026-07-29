"""Paper Fig 1 (fig:inclusive_data): inclusive (e,e') d2sigma/dOmega dE on 40Ar (left) and 12C (right)
at E_beam=2.222 GeV, theta_e'=15.541 deg, ADoNIS vs ACHILLES with the QE / RES / TOTAL decomposition.

Mirrors the paper: Ar-left / C-right, omega [GeV] on x, QE (blue) + RES (green) + TOTAL (black) for both
generators (ADoNIS solid, ACHILLES dashed).  omega is lepton-side (FSI-independent), so the ADoNIS beam_e
bank ([5,45], cut to [14,17]) compares to the ACHILLES inclusive-ee oracle (no cascade) at [14,17].
ADoNIS: bank omega/theta/c + channel (0=QE, 1=RES).  ACHILLES: inclusive_ee_{nuc}_{qe,res}, lep->omega.

  python -m analysis.paper.sec1_validation.fig1_ee_domega
"""
import sys
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.workflow.plotting import chi2_ratio_panel     # noqa: E402
from analysis.paper import style                          # noqa: E402

EB = 2222.0
THE_LO, THE_HI = 14.0, 17.0
NUC_TEX = {"C": r"$^{12}$C", "Ar": r"$^{40}$Ar"}
EDGES = np.linspace(0.05, 0.95, 46)                       # omega [GeV]  (paper x-range)


def adonis(nuc):
    """omega[GeV], w, channel for the beam_e bank cut to [14,17] deg."""
    bd = ROOT / f"output/paper_banks_p4/beam_e_{nuc}/merged"
    nch = json.load(open(bd / "manifest.json"))["n_chunks"]
    O, W, CH = [], [], []
    for f in sorted(glob.glob(str(bd / "chunk_*.npz"))):
        d = np.load(f); th = np.asarray(d["theta"], float); k = (th >= THE_LO) & (th <= THE_HI)
        O.append(np.asarray(d["omega"], float)[k] / 1000.0)          # MeV -> GeV
        W.append(np.asarray(d["c"], float)[k] / nch)
        CH.append(np.asarray(d["channel"])[k])
    return np.concatenate(O), np.concatenate(W), np.concatenate(CH)


def achilles(nuc, ch):
    """omega[GeV], w for one channel (qe|res) inclusive-ee oracle, cut to [14,17] deg."""
    d = np.load(ROOT / f"output/achilles/fsrich/inclusive_ee_{nuc}_{ch}.npz", allow_pickle=True)
    lep = np.asarray(d["lep"], float); Ee = lep[:, 0]
    pe = np.linalg.norm(lep[:, 1:], axis=1)
    cth = np.where(pe > 0, lep[:, 3] / np.maximum(pe, 1e-9), -2.0)
    the = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    k = (the >= THE_LO) & (the <= THE_HI)
    w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
    return (EB - Ee)[k] / 1000.0, w[k]


def main():
    style.use()
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 5.2), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.24})
    for ni, nuc in enumerate(("Ar", "C")):                # paper order: Ar left, C right
        ao, aw, ach_lab = adonis(nuc)
        aqe = ach_lab == 0; ares = ach_lab == 1
        hqo, hqw = achilles(nuc, "qe"); hro, hrw = achilles(nuc, "res")
        ho = np.concatenate([hqo, hro]); hw = np.concatenate([hqw, hrw])
        res = chi2_ratio_panel(
            ax[0, ni], ax[1, ni], EDGES,
            {"values": ho, "w": hw}, {"values": ao, "w": aw},
            label=rf"(e,e') {NUC_TEX[nuc]}", xlabel=r"$\omega$ [GeV]", ratio_ylim=(0.6, 1.4),
            ado_label="ADoNIS", ref_label="ACHILLES",
            ref_parts={"QE": {"values": hqo, "w": hqw}, "RES": {"values": hro, "w": hrw}},
            ado_parts={"QE": {"values": ao[aqe], "w": aw[aqe]},
                       "RES": {"values": ao[ares], "w": aw[ares]}})
        print(f"  {nuc} chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} (ndf={res['ndf']}) "
              f"nADO={len(aw)} nACH={len(hw)}", flush=True)
        if ni == 0:
            ax[0, ni].legend(fontsize=6, ncol=2); ax[0, ni].set_ylabel(r"$d\sigma/d\omega$ [nb/GeV]")
            ax[1, ni].set_ylabel("ratio")
    fig.suptitle(r"ADoNIS vs ACHILLES --- inclusive (e,e') at 2.222 GeV, "
                 r"$\theta_{e'}\approx15.5^\circ$  (QE / RES / total)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig01_ee_domega")


if __name__ == "__main__":
    main()
