"""Paper Fig 1: inclusive (e,e') dsigma/domega on 12C and 40Ar at 2.222 GeV, theta_e ~ 15.5 deg
([14,17] window), ADoNIS vs ACHILLES.

omega (energy transfer) is a lepton-side observable, so it's FSI-independent -- the ADoNIS beam_e bank
(fsi, [5,45]) cut to [14,17] compares to the ACHILLES inclusive-ee oracle (no cascade) at [14,17].
ADoNIS: omega/theta/c.  ACHILLES: inclusive_ee_{nuc}_{qe,res} (lep -> omega = E_beam - E_e').

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
EDGES = np.linspace(0.0, 800.0, 41)                        # omega [MeV]


def adonis_omega(nuc):
    bd = ROOT / f"output/paper_banks_p4/beam_e_{nuc}/merged"
    nch = json.load(open(bd / "manifest.json"))["n_chunks"]
    O, W = [], []
    for f in sorted(glob.glob(str(bd / "chunk_*.npz"))):
        d = np.load(f); th = np.asarray(d["theta"], float); k = (th >= THE_LO) & (th <= THE_HI)
        O.append(np.asarray(d["omega"], float)[k]); W.append(np.asarray(d["c"], float)[k] / nch)
    return np.concatenate(O), np.concatenate(W)


def achilles_omega(nuc):
    O, W = [], []
    for ch in ("qe", "res"):
        d = np.load(ROOT / f"output/achilles/fsrich/inclusive_ee_{nuc}_{ch}.npz", allow_pickle=True)
        lep = np.asarray(d["lep"], float); Ee = lep[:, 0]
        pe = np.linalg.norm(lep[:, 1:], axis=1)
        cth = np.where(pe > 0, lep[:, 3] / np.maximum(pe, 1e-9), -2.0)
        the = np.degrees(np.arccos(np.clip(cth, -1, 1)))
        k = (the >= THE_LO) & (the <= THE_HI)
        w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
        O.append((EB - Ee)[k]); W.append(w[k])
    return np.concatenate(O), np.concatenate(W)


def main():
    style.use()
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 5.0), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.26})
    for ni, nuc in enumerate(("C", "Ar")):
        ao, aw = adonis_omega(nuc); ho, hw = achilles_omega(nuc)
        res = chi2_ratio_panel(ax[0, ni], ax[1, ni], EDGES, {"values": ho, "w": hw},
                               {"values": ao, "w": aw}, label=rf"(e,e') {NUC_TEX[nuc]}: $\omega$ [MeV]",
                               ratio_ylim=(0.6, 1.4), ado_label="ADoNIS", ref_label="ACHILLES")
        print(f"  {nuc} chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} (ndf={res['ndf']}) "
              f"nADO={len(aw)} nACH={len(hw)}", flush=True)
        if ni == 0:
            ax[0, ni].legend(fontsize=7); ax[0, ni].set_ylabel(r"$d\sigma/d\omega$ [nb/MeV]")
            ax[1, ni].set_ylabel("ratio")
    fig.suptitle(r"ADoNIS vs ACHILLES --- inclusive (e,e') $d\sigma/d\omega$ at 2.222 GeV, "
                 r"$\theta_e\approx15.5^\circ$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig01_ee_domega")


if __name__ == "__main__":
    main()
