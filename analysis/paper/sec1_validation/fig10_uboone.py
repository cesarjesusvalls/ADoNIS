"""Paper Fig 10: MicroBooNE CC1p0pi double-differential dsigma/(d delta_pT, d delta_alphaT) on 40Ar --
delta_pT distributions in 4 delta_alphaT slices, ADoNIS vs ACHILLES.

Signal + binning from the NUISANCE data release (MicroBooNE/CC1Mu1p, Abratenko et al. 2023):
  mu 100-1200 MeV/c, exactly ONE proton 300-1000 MeV/c, 0 pions, cos(theta) full range.
  delta_alphaT slices: [0,45), [45,90), [90,135), [135,180] deg; delta_pT bins per slice (11/12/13/13).
The 4 slices are 4 delta_pT histograms through the shared chi2_ratio_panel (histogram mode).

  python -m analysis.paper.sec1_validation.fig10_uboone
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.workflow.config import SignalDef              # noqa: E402
from adonis.workflow import selection as SG               # noqa: E402
from adonis.workflow.plotting import chi2_ratio_panel     # noqa: E402
from analysis.paper import style                          # noqa: E402

BANK = str(ROOT / "output/paper_banks_p4/nu_uBooNE_Ar/merged")
ORA = str(ROOT / "output/achilles/fsrich/nu_uBooNE_Ar.npz")

SD = SignalDef(mu_win=(100.0, 1200.0), cos_mu=-1.0, p_win=(300.0, 1000.0), cth=-1.0,
               pi_win=None, pion_id="none", proton_lead="global", proton_count="eq1")

SLICES = [(0, 45), (45, 90), (90, 135), (135, 180)]        # delta_alphaT [deg]
DPT_EDGES = [                                              # delta_pT bin edges per slice [GeV/c] (NUISANCE)
    [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.47, 0.55, 0.90],
    [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.47, 0.55, 0.65, 0.90],
    [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.47, 0.55, 0.65, 0.75, 0.90],
    [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.47, 0.55, 0.65, 0.75, 0.90],
]


def main():
    style.use()
    ado = SG.bank_signal(BANK, SD); ach = SG.oracle_signal(ORA, SD)
    print(f"CC1p0pi selected: nADO={len(ado['w'])} nACH={len(ach['w'])}", flush=True)
    dat_a = np.degrees(ado["dalphat"]); dat_h = np.degrees(ach["dalphat"])
    fig, ax = plt.subplots(2, 4, figsize=(15.0, 5.4), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.30})
    for si, (lo, hi) in enumerate(SLICES):
        edges = np.asarray(DPT_EDGES[si]) * 1000.0            # GeV -> MeV (dpt is in MeV)
        ma = (dat_a >= lo) & (dat_a < hi); mh = (dat_h >= lo) & (dat_h < hi)
        res = chi2_ratio_panel(ax[0, si], ax[1, si], edges,
                               {"values": ach["dpt"][mh], "w": ach["w"][mh]},
                               {"values": ado["dpt"][ma], "w": ado["w"][ma]},
                               label=rf"$\delta\alpha_T\in[{lo},{hi})^\circ$",
                               xlabel=r"$\delta p_T$ [MeV/c]", ratio_ylim=(0.5, 1.5),
                               ado_label="ADoNIS", ref_label="ACHILLES")
        print(f"  slice [{lo:3d},{hi:3d}) chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} "
              f"(ndf={res['ndf']}) nADO={int(ma.sum())} nACH={int(mh.sum())}", flush=True)
        if si == 0:
            ax[0, si].legend(fontsize=7); ax[0, si].set_ylabel(r"$d\sigma/d\delta p_T$ [nb]")
            ax[1, si].set_ylabel("ratio")
    fig.suptitle(r"ADoNIS vs ACHILLES --- MicroBooNE CC1p0$\pi$: $\delta p_T$ in $\delta\alpha_T$ "
                 r"slices on $^{40}$Ar", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    style.save(fig, "fig10_uboone_cc1p0pi")


if __name__ == "__main__":
    main()
