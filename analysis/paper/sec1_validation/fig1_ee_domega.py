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
from matplotlib.lines import Line2D

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


def _legend(a):
    """Factorised legend: one entry per component (colour), then ONE grey 'dashed / solid' swatch for
    the generator (linestyle).  6 entries with repeated 'ACHILLES ...'/'ADoNIS ...' strings collapse to
    4, and the two encodings are named separately instead of multiplied out."""
    # No generator key entry: dashed=ACHILLES / solid=ADoNIS is stated in the caption.
    handles, labels, hmap = style.swatches(
        [("Total", style.C_TOTAL), ("QE", style.C_QE), ("RES", style.C_RES)])
    # anchored left of the corner: without the long generator row the box is narrow, so a plain
    # "upper right" puts "Total" under the nucleus tag.
    a.legend(handles, labels, handler_map=hmap,
             loc="upper right", bbox_to_anchor=(0.84, 1.0), fontsize=7, ncol=1,
             handlelength=3.4, labelspacing=0.35, borderpad=0.2)


def main():
    style.use()
    # Smaller canvas at unchanged absolute font sizes -> text reads larger relative to the axes.
    # Height: both panels halved (3:1 kept), so only the ~0.85in of title/label/tick overhead survives.
    fig, ax = plt.subplots(2, 2, figsize=(7.1, 2.6), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.26})
    for ni, nuc in enumerate(("Ar", "C")):                # paper order: Ar left, C right
        ao, aw, ach_lab = adonis(nuc)
        aqe = ach_lab == 0; ares = ach_lab == 1
        hqo, hqw = achilles(nuc, "qe"); hro, hrw = achilles(nuc, "res")
        ho = np.concatenate([hqo, hro]); hw = np.concatenate([hqw, hrw])
        res = chi2_ratio_panel(
            ax[0, ni], ax[1, ni], EDGES,
            {"values": ho, "w": hw}, {"values": ao, "w": aw},
            label="", xlabel=r"$\omega$ [GeV]", ratio_ylim=(0.6, 1.4),   # nucleus goes IN the axes
            ado_label="ADoNIS", ref_label="ACHILLES",
            total_color=style.C_TOTAL, part_colors=style.PARTS, ratio_color=style.C_RATIO,
            ado_lighten=style.ADO_LIGHTEN, ref_darken=style.REF_DARKEN, headroom=0.30,
            ref_parts={"QE": {"values": hqo, "w": hqw}, "RES": {"values": hro, "w": hrw}},
            ado_parts={"QE": {"values": ao[aqe], "w": aw[aqe]},
                       "RES": {"values": ao[ares], "w": aw[ares]}})
        print(f"  {nuc} chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} (ndf={res['ndf']}) "
              f"nADO={len(aw)} nACH={len(hw)}", flush=True)
        # nucleus label inside the panel (an axes title collides with the suptitle on a short canvas):
        # top-RIGHT corner, with the legend anchored just below it -- the falling tail leaves that side free.
        ax[0, ni].text(0.975, 0.95, NUC_TEX[nuc], transform=ax[0, ni].transAxes,
                       fontsize=9, va="top", ha="right")
        if ni == 0:
            _legend(ax[0, ni]); ax[0, ni].set_ylabel(r"$d\sigma/d\omega$ [nb/GeV]")
            ax[1, ni].set_ylabel("ratio")
    fig.suptitle(r"Inclusive (e,e') at 2.222 GeV, $\theta_{e'}\approx15.5^\circ$", fontsize=9, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig01_ee_domega")


if __name__ == "__main__":
    main()
