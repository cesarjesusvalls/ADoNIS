"""dsigma/d|p| of the ejected NEUTRON by RANK (0=leading, 1=sub-leading, ...), ADoNIS vs ACHILLES.
Banks store the top-4 neutrons per event sorted by |p| (ADoNIS `neut`; ACHILLES `neut_p4` padded).
Muon-acceptance CC-inclusive, absolute dsigma.  Run: CHANNEL=qe TAG=run python -u scripts/neutron_rank_kin.py out.png
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.plotting import chi2_ratio_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/neutron_rank_kin.png"
CH = os.environ.get("CHANNEL", "qe").lower(); TAG = os.environ.get("TAG", "run")
ACH = os.environ["ACH_BANK"]; PROC = [200] if CH == "qe" else [401, 402]
GLOB = f"data/oracle/t2k_{'cc0pi' if CH=='qe' else 'cc1pi'}_{TAG}_batch*.npz"
COS70 = float(np.cos(np.deg2rad(70.0))); NR = 4; EDGES = np.linspace(0, 1000, 41)


def mu(m):
    p = np.linalg.norm(m[:, 1:], axis=1); cz = m[:, 3] / np.clip(p, 1e-9, None)
    return (p > 250) & (p < 7000) & (cz > COS70)


def load_ado():
    fs = [f for f in sorted(glob.glob(GLOB)) if "neut" in np.load(f).files]; nb = len(fs); P = []; W = []
    for f in fs:
        b = dict(np.load(f, allow_pickle=True)); m = mu(b["mu"]) & (b["w"] > 0)
        P.append(np.linalg.norm(b["neut"][m][:, :, 1:], axis=2)); W.append(b["w"][m] / nb)
    return np.concatenate(P, 0), np.concatenate(W)


def load_ach():
    d = dict(np.load(ACH, allow_pickle=True)); sel = np.isin(d["proc"], PROC) & mu(d["mu"])
    return np.linalg.norm(d["neut_p4"][sel][:, :, 1:], axis=2), d["w"][sel] * float(d["weight_to_nb"])


def main():
    A, wa = load_ado(); H, wh = load_ach()
    fig, ax = plt.subplots(2, NR, figsize=(4.6 * NR, 6.5), height_ratios=[3, 1], sharex="col")
    names = ["leading", "sub-leading", "sub-sub", "rank-3"]
    for rk in range(NR):
        pa = A[:, rk]; ph = H[:, rk] if rk < H.shape[1] else np.zeros(0)
        ma = pa > 0; mh = ph > 0
        a0, a1 = ax[0, rk], ax[1, rk]; a0.set_xlim(EDGES[0], EDGES[-1])
        chi2_ratio_panel(a0, a1, EDGES, {"values": ph[mh], "w": wh[mh]}, {"values": pa[ma], "w": wa[ma]},
                         label=f"neutron {names[rk]}", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
        sa = wa[ma].sum(); sh = wh[mh].sum()
        a0.text(0.5, 0.86, f"σ ACH/ADO={sh/max(sa,1e-30):.3f}", transform=a0.transAxes, ha="center",
                fontsize=9, color="purple")
    ax[0, 0].set_ylabel("dσ/d|p| [nb/MeV]"); ax[1, 0].set_ylabel("ACH/ADO"); ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{CH.upper()} ejected-neutron |p| by rank (P=1, {TAG}) vs ACHILLES")
    fig.tight_layout(); fig.savefig(OUT, dpi=115); print(f"wrote {OUT}", flush=True)
    for rk in range(NR):
        pa = A[:, rk]; ph = H[:, rk] if rk < H.shape[1] else np.zeros(0)
        print(f"  neutron rank {rk}: ADO σ={wa[pa>0].sum():.3e} (N={int((pa>0).sum())})  "
              f"ACH σ={wh[ph>0].sum():.3e} (N={int((ph>0).sum())})  median|p| ADO={np.median(pa[pa>0]) if (pa>0).any() else 0:.0f} "
              f"ACH={np.median(ph[ph>0]) if (ph>0).any() else 0:.0f} MeV", flush=True)


if __name__ == "__main__":
    main()
