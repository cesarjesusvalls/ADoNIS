"""Per-nucleon-RANK momentum agreement: dsigma/dp for the leading, sub-leading, sub-sub-leading...
proton (and neutron), ADoNIS vs ACHILLES.  Localizes the busy-event multi-knockout deficit -- if the
cascade is 'too quiet' in busy events, the deficit grows with rank (leading fine, higher ranks low).

ADoNIS banks: prot/neut = top-mprot (=4) nucleon 4-vecs sorted by |p| (rank 0=leading).  ACHILLES
oracle: prot_p4/neut_p4 (padded 10, sorted by |p|).  Muon-acceptance CC-inclusive, absolute dsigma.

Run: CHANNEL=qe ACH_BANK=...proc.npz python -u scripts/rank_momentum.py <out.png>
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.plotting import chi2_ratio_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rank_momentum.png"
CH = os.environ.get("CHANNEL", "qe").lower()
ADO_DIR = os.environ.get("ADO_DIR", "data/oracle"); ACH = os.environ["ACH_BANK"]
PROC = [200] if CH == "qe" else [401, 402]
TAG = os.environ.get("TAG", "cv5")                                     # cv5=C pre-fix, cfix=C post-fix, av5=Ar
ADO_GLOB = f"{ADO_DIR}/t2k_{'cc0pi' if CH=='qe' else 'cc1pi'}_engine_rich_{TAG}_batch*.npz"
COS70 = float(np.cos(np.deg2rad(70.0))); NRANK = 4; EDGES = np.linspace(0, 1200, 25)


def mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > 250) & (p < 7000) & (cz > COS70)


def load_ado():
    fs = [f for f in sorted(glob.glob(ADO_GLOB)) if "neut" in np.load(f).files]; nb = len(fs)
    P = {0: [], 1: []}; W = []   # 0=proton(field 'prot'), 1=neutron(field 'neut')
    for f in fs:
        b = dict(np.load(f, allow_pickle=True)); m = mu_mask(b["mu"]) & (b["w"] > 0)
        W.append(b["w"][m] / nb)
        P[0].append(np.linalg.norm(b["prot"][m][:, :, 1:], axis=2))   # (n, mprot) |p| per rank
        P[1].append(np.linalg.norm(b["neut"][m][:, :, 1:], axis=2))
    return {k: np.concatenate(v, 0) for k, v in P.items()}, np.concatenate(W)


def load_ach():
    d = dict(np.load(ACH, allow_pickle=True)); sel = np.isin(d["proc"], PROC) & mu_mask(d["mu"])
    w = d["w"][sel] * float(d["weight_to_nb"])
    P = {0: np.linalg.norm(d["prot_p4"][sel][:, :, 1:], axis=2),
         1: np.linalg.norm(d["neut_p4"][sel][:, :, 1:], axis=2)}
    return P, w


def main():
    A, wa = load_ado(); H, wh = load_ach()
    names = {0: "proton", 1: "neutron"}
    fig, ax = plt.subplots(4, NRANK, figsize=(4.6 * NRANK, 12), height_ratios=[3, 1, 3, 1], sharex="col")
    for sp in (0, 1):
        r0, r1 = (0, 1) if sp == 0 else (2, 3)
        for rk in range(NRANK):
            pa = A[sp][:, rk]; ph = H[sp][:, rk] if rk < H[sp].shape[1] else np.zeros(0)
            ma = pa > 0; mh = ph > 0
            a0, a1 = ax[r0, rk], ax[r1, rk]; a0.set_xlim(EDGES[0], EDGES[-1])
            chi2_ratio_panel(a0, a1, EDGES, {"values": ph[mh], "w": wh[mh]},
                             {"values": pa[ma], "w": wa[ma]},
                             label=f"{names[sp]} rank {rk}", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
            # annotate the integrated ACH/ADO for this rank (= sigma of events with >=rk+1 of this species)
            sa = wa[ma].sum(); sh = wh[mh].sum()
            a0.text(0.5, 0.88, f"σ ACH/ADO={sh/max(sa,1e-30):.3f}", transform=a0.transAxes,
                    ha="center", fontsize=8, color="purple")
    for sp, r in ((0, 0), (1, 2)):
        ax[r, 0].set_ylabel(f"{names[sp]}  dσ/dp [nb/MeV]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"C-{CH.upper()}: dσ/dp per nucleon RANK (0=leading,1=sub,2=sub-sub...), ADoNIS vs ACHILLES "
                 f"(mass-fixed, Gaussian)")
    fig.tight_layout(); fig.savefig(OUT, dpi=110); print(f"wrote {OUT}", flush=True)
    for sp in (0, 1):
        print(f"  {names[sp]}: integrated σ ACH/ADO by rank:", flush=True)
        for rk in range(NRANK):
            pa = A[sp][:, rk]; ph = H[sp][:, rk] if rk < H[sp].shape[1] else np.zeros(0)
            sa = wa[pa > 0].sum(); sh = wh[ph > 0].sum()
            print(f"    rank {rk}: ADO={sa:.3e}  ACH={sh:.3e}  ACH/ADO={sh/max(sa,1e-30):.3f}", flush=True)


if __name__ == "__main__":
    main()
