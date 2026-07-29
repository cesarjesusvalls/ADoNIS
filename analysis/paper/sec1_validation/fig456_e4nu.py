"""Paper Figs 4/5/6: e4nu (e,e') on 12C at 1.159 GeV, ADoNIS vs ACHILLES.
  Fig 4: dsigma/dE_QE   (0pi)     -- reconstructed quasi-elastic energy
  Fig 5: dsigma/dE_cal  (1p0pi)   -- calorimetric energy  E_cal = E_e' + T_p + eps
  Fig 6: dsigma/dP_T    (1p0pi)   -- transverse momentum imbalance |p_T^e' + p_T^p|
e4nu/CLAS6 (Khachatryan 2021): electron 15<=theta_e<=45 deg & E_e'>=0.4 GeV; proton p>0.3 GeV/c,
10<=theta_p<=140 deg; 0 pions (1p0pi requires exactly one proton in acceptance).

ADoNIS: EM bank stores omega/theta/c (E_e'=E_beam-omega; electron azimuth taken =0, the scattering
plane -> validated vs the oracle P_T). ACHILLES: ee_C_1159 qe+res fs_rich (full lep 4-vector).

  python -m analysis.paper.sec1_validation.fig456_e4nu
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.workflow.plotting import chi2_ratio_panel      # noqa: E402
from analysis.paper import style                           # noqa: E402

EB = 1159.0; MNUC = 938.9; MP = 938.272; ME = 0.511; EPS = 21.0
PP_MIN = 300.0; TP_LO, TP_HI = 10.0, 140.0                 # proton acceptance
EE_MIN, THE_LO, THE_HI = 400.0, 15.0, 45.0                 # electron acceptance
BANK = str(ROOT / "output/paper_banks_p4/beam_e_C_1159/merged")
ORA = [str(ROOT / f"output/achilles/fsrich/ee_C_1159_{c}_fsi.npz") for c in ("qe", "res")]


def _eqe(Ee, cth):
    pe = np.sqrt(np.maximum(Ee ** 2 - ME ** 2, 0.0))
    return (2 * MNUC * EPS + 2 * MNUC * Ee - ME ** 2) / (2 * (MNUC - Ee + pe * cth))


def _lead_proton_seg(pid, p4, seg, n):
    """leading in-acceptance proton per event (n,4) + count of in-acc protons per event."""
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    cth = np.where(mom > 0, p4[:, 3] / np.maximum(mom, 1e-9), -2.0)
    th = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    acc = (pid == 2212) & (mom > PP_MIN) & (th >= TP_LO) & (th <= TP_HI)
    nprot = np.zeros(n); np.add.at(nprot, seg[acc], 1.0)
    key = np.where(acc, mom, -1.0); mx = np.full(n, -1.0); np.maximum.at(mx, seg, key)
    lead = np.zeros((n, 4)); islead = acc & (key == mx[seg]) & (mom > 0)
    lead[seg[islead]] = p4[islead]
    return lead, nprot.astype(int)


def adonis_obs():
    files = sorted(glob.glob(BANK + "/chunk_*.npz"))
    man = __import__("json").load(open(Path(BANK) / "manifest.json")); nch = man["n_chunks"]
    EQ, wq, EC, PT, wc = [], [], [], [], []
    for f in files:
        d = np.load(f)
        c = np.asarray(d["c"], float) / nch                  # (e,e') per-event weight (averaged like nu)
        om = np.asarray(d["omega"], float); th = np.radians(np.asarray(d["theta"], float))
        Ee = EB - om; pe = np.sqrt(np.maximum(Ee ** 2 - ME ** 2, 0.0)); cth = np.cos(th)
        elec = (Ee >= EE_MIN) & (np.degrees(th) >= THE_LO) & (np.degrees(th) <= THE_HI)
        npi = np.asarray(d["n_pi_out"])
        # Fig 4: E_QE, 0pi
        k0 = elec & (npi == 0)
        EQ.append(_eqe(Ee, cth)[k0]); wq.append(c[k0])
        # leading proton for 1p0pi (Figs 5/6)
        ne = len(c); cnt = np.diff(np.asarray(d["fs_off"], np.int64))
        seg = np.repeat(np.arange(ne), cnt)
        lead, nprot = _lead_proton_seg(np.asarray(d["fs_pid"]), np.asarray(d["fs_p4"], float), seg, ne)
        k1 = elec & (npi == 0) & (nprot == 1)
        Tp = lead[:, 0] - MP                                  # proton KE
        EC.append((Ee + Tp + EPS)[k1]); wc.append(c[k1])
        ke = np.asarray(d["k_e"], float)                     # outgoing e- 4-vector (full azimuth)
        pt = np.sqrt((ke[:, 1] + lead[:, 1]) ** 2 + (ke[:, 2] + lead[:, 2]) ** 2)
        PT.append(pt[k1])
    return (np.concatenate(EQ), np.concatenate(wq)), (np.concatenate(EC), np.concatenate(wc),
            np.concatenate(PT))


def achilles_obs():
    EQ, wq, EC, PT, wc = [], [], [], [], []
    for p in ORA:
        d = np.load(p, allow_pickle=True)
        w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
        lep = np.asarray(d["lep"], float); Ee = lep[:, 0]
        pe = np.linalg.norm(lep[:, 1:], axis=1); cth = np.where(pe > 0, lep[:, 3] / np.maximum(pe, 1e-9), -2.0)
        the = np.degrees(np.arccos(np.clip(cth, -1, 1)))
        elec = (Ee >= EE_MIN) & (the >= THE_LO) & (the <= THE_HI)
        pipid = np.asarray(d["pi_pid"]); npi = (pipid != 0).sum(1)
        k0 = elec & (npi == 0)
        EQ.append(_eqe(Ee, cth)[k0]); wq.append(w[k0])
        prot = np.asarray(d["prot_p4"], float); pm = np.linalg.norm(prot[:, :, 1:], axis=2)
        pcz = np.where(pm > 0, prot[:, :, 3] / np.maximum(pm, 1e-9), -2.0)
        tp = np.degrees(np.arccos(np.clip(pcz, -1, 1)))
        acc = (pm > PP_MIN) & (tp >= TP_LO) & (tp <= TP_HI)
        nprot = acc.sum(1)
        key = np.where(acc, pm, -1.0); j = key.argmax(1); lead = prot[np.arange(len(lep)), j]
        k1 = elec & (npi == 0) & (nprot == 1)
        Tp = lead[:, 0] - MP
        EC.append((Ee + Tp + EPS)[k1]); wc.append(w[k1])
        pt = np.sqrt((lep[:, 1] + lead[:, 1]) ** 2 + (lep[:, 2] + lead[:, 2]) ** 2)
        PT.append(pt[k1])
    return (np.concatenate(EQ), np.concatenate(wq)), (np.concatenate(EC), np.concatenate(wc),
            np.concatenate(PT))


def main():
    style.use()
    (aEQ, awq), (aEC, awc, aPT) = adonis_obs()
    (hEQ, hwq), (hEC, hwc, hPT) = achilles_obs()
    print(f"0pi: nADO={len(awq)} nACH={len(hwq)} | 1p0pi: nADO={len(awc)} nACH={len(hwc)}", flush=True)
    panels = [("E_QE [MeV]  (0$\\pi$)", np.linspace(600, 1300, 25), aEQ, awq, hEQ, hwq),
              ("E_cal [MeV]  (1p0$\\pi$)", np.linspace(700, 1300, 25), aEC, awc, hEC, hwc),
              ("$P_T$ [MeV/c]  (1p0$\\pi$)", np.linspace(0, 600, 25), aPT, awc, hPT, hwc)]
    fig, ax = plt.subplots(2, 3, figsize=(12, 5.2), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0, "wspace": 0.26})
    for c, (lab, edges, av, aw, hv, hw) in enumerate(panels):
        res = chi2_ratio_panel(ax[0, c], ax[1, c], edges, {"values": hv, "w": hw}, {"values": av, "w": aw},
                               label=lab, ratio_ylim=(0.6, 1.4), ado_label="ADoNIS", ref_label="ACHILLES")
        print(f"  {lab[:12]:14s} chi2/ndf {res['chi2']/max(res['ndf'],1):6.2f} (ndf={res['ndf']})", flush=True)
        if c == 0:
            ax[0, c].legend(fontsize=7); ax[0, c].set_ylabel(r"$d\sigma/dx$ [nb]"); ax[1, c].set_ylabel("ratio")
    fig.suptitle(r"ADoNIS vs ACHILLES --- e4$\nu$ (e,e') on $^{12}$C at 1.159 GeV", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    style.save(fig, "fig456_e4nu")


if __name__ == "__main__":
    main()
