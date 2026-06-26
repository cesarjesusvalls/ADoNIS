"""RES multiplicity dsigma vs ACHILLES, split by the 3 primary CC-1pi channels:
  pp   : p -> p pi+   (struck p, primary recoil p)
  npi0 : n -> p pi0   (struck n, primary recoil p)
  npip : n -> n pi+   (struck n, primary recoil n)
ADoNIS channel = (ipid struck, Npid primary recoil) from the bank.  ACHILLES channel = (struck_pid,
primary_recoil_pid) -- the latter extracted from the HepMC hard vertex (sidecar primrecoil.npz).
Same absolute, muon-acceptance comparison as multiplicity_xsec.py.

Run: RESCHAN=npip TAG=run python -u scripts/multiplicity_reschan.py /tmp/out.png
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.plotting import chi2_ratio_panel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mult_reschan.png"
TAG = os.environ.get("TAG", "run")
CHAN = os.environ.get("RESCHAN", "npip")                 # pp | npi0 | npip
ACH = os.environ.get("ACH_BANK", "data/oracle/t2k_C_res_gauss_proc.npz")
SIDE = os.environ.get("ACH_SIDE", "data/oracle/t2k_C_res_gauss_primrecoil.npz")
MU_WIN = [250.0, 7000.0]; COS70 = float(np.cos(np.deg2rad(70.0)))
SPECIES = ["N(p)", "N(n)", "N(pi+)", "N(pi0)", "N(pi-)"]
FIELD = {"N(p)": "n_p", "N(n)": "n_n", "N(pi+)": "n_pip", "N(pi0)": "n_pi0", "N(pi-)": "n_pim"}
# channel -> (struck pid, primary recoil pid)
CH_DEF = {"pp": (2212, 2212), "npi0": (2112, 2212), "npip": (2112, 2112)}
CH_TITLE = {"pp": "p -> p pi+", "npi0": "n -> p pi0", "npip": "n -> n pi+"}
struck_c, recoil_c = CH_DEF[CHAN]


def mu_mask(mu):
    p = np.linalg.norm(mu[:, 1:], axis=1); cz = mu[:, 3] / np.clip(p, 1e-9, None)
    return (p > MU_WIN[0]) & (p < MU_WIN[1]) & (cz > COS70)


def ado():
    fs = [f for f in sorted(glob.glob(f"data/oracle/t2k_cc1pi_engine_rich_{TAG}_batch*.npz"))
          if "n_p" in np.load(f, allow_pickle=True).files]
    nb = len(fs); cnt = {s: [] for s in SPECIES}; ws = []
    for p in fs:
        b = dict(np.load(p, allow_pickle=True))
        chan = (b["ipid"] == struck_c) & (b["Npid"] == recoil_c)
        m = mu_mask(b["mu"]) & (b["w"] > 0) & chan
        for s in SPECIES:
            cnt[s].append(b[FIELD[s]][m])
        ws.append(b["w"][m])
    return {s: np.concatenate(cnt[s]) for s in SPECIES}, np.concatenate(ws) / nb, nb


def ach():
    d = dict(np.load(ACH, allow_pickle=True)); recoil = np.load(SIDE)["primary_recoil_pid"]
    chan = (d["struck_pid"] == struck_c) & (recoil == recoil_c)
    sel = np.isin(d["proc"], [401, 402]) & mu_mask(d["mu"]) & chan
    w = d["w"][sel] * float(d["weight_to_nb"])
    pp4 = d["prot_p4"][sel]; np4 = d["neut_p4"][sel]; pip = d["pi_pid"][sel]
    out = {"N(p)": (np.linalg.norm(pp4[:, :, 1:], axis=2) > 1e-6).sum(1),
           "N(n)": (np.linalg.norm(np4[:, :, 1:], axis=2) > 1e-6).sum(1),
           "N(pi+)": (pip == 211).sum(1), "N(pi0)": (pip == 111).sum(1), "N(pi-)": (pip == -211).sum(1)}
    return out, w


def main():
    aQ, wQ, nb = ado(); hQ, hwQ = ach()
    nmax = 6; edges = np.arange(nmax + 2.0)
    fig, axes = plt.subplots(2, 5, figsize=(23, 6), height_ratios=[3, 1], sharex="col")
    for c, lab in enumerate(SPECIES):
        a0, a1 = axes[0, c], axes[1, c]; a0.set_xlim(edges[0], edges[-1])
        chi2_ratio_panel(a0, a1, edges, {"values": hQ[lab], "w": hwQ}, {"values": aQ[lab], "w": wQ},
                         label=f"{CHAN}: {lab}", ado_label="ADoNIS", ref_label="ACHILLES", logy=True)
    axes[0, 0].set_ylabel("dσ/dN [nb]"); axes[1, 0].set_ylabel("ACH/ADO"); axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"RES channel {CH_TITLE[CHAN]}  ({CHAN}): dσ vs multiplicity, ADoNIS(P=1) vs ACHILLES  "
                 f"[ADO σ={wQ.sum():.3e}  ACH σ={hwQ.sum():.3e}  ACH/ADO={hwQ.sum()/max(wQ.sum(),1e-30):.3f}]")
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT}", flush=True)
    print(f"--- {CHAN} ({CH_TITLE[CHAN]}): ADO σ={wQ.sum():.4e}  ACH σ={hwQ.sum():.4e}  ACH/ADO={hwQ.sum()/wQ.sum():.3f}", flush=True)
    for lab in SPECIES:
        a = np.array([wQ[np.clip(aQ[lab], 0, 6) == k].sum() for k in range(4)])
        h = np.array([hwQ[np.clip(hQ[lab], 0, 6) == k].sum() for k in range(4)])
        print(f"  {lab:7s} ADO N0..3: {a}   ACH: {h}", flush=True)


if __name__ == "__main__":
    main()
