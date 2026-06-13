"""SUBSTITUTION-BISECT of the CC1pi high-pi_p/high-W tail: apply the IDENTICAL signal cut ladder
to the ACHILLES res_w_FSI bank and the ADoNIS res_C raw bank, and localize which cut opens the
ADO-high tail. The cut ladder (carbon both sides):
   stage 0  base    = mu-acc + leading pi+ in window & forward      (== the pion-survival selection)
   stage 1  +1pi    = + exactly one pion         (ACH n_pi==1 ; ADO no_extra_pi)
   stage 2  +prot   = + leading in-window proton  (ACH prot_ok ; ADO has_p)   == FULL signal
Key diagnostic: P(in-window proton | pi_p) and | W -- the proton-leg acceptance conditional.
Pure re-binning of cached banks; no generation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.data.oracle.normalization import hepmc_norm
COS70 = np.cos(np.deg2rad(70.0))


def ach_masks():
    a = np.load("data/oracle/t2k_res_w_achilles_FSI.npz")
    w = a["w"].astype(float) * hepmc_norm("_oracle_out/T2K_CH_virt.hepmc")["weight_to_nb"]
    base = ((a["pstr"] > 1.0) & (a["mu_p"] > 250) & (a["mu_p"] < 7000) & (a["mu_cth"] > COS70)
            & (a["pi_pid"] == 211) & (a["pi_p"] > 150) & (a["pi_p"] < 1200) & (a["pi_cth"] > COS70))
    one = base & (a["n_pi"] == 1)
    sig = one & a["prot_ok"]
    return dict(pi_p=a["pi_p"], W=a["W"], w=w, base=base, one=one, sig=sig, prot=a["prot_ok"].astype(bool))


def ado_masks():
    a = np.load("data/oracle/t2k_cc1pi_raw_adonis.npz")
    w = a["w"].astype(float)
    base = ((a["pid_pi"] == 211) & (w > 0) & (a["mu_p"] > 250) & (a["mu_p"] < 7000) & (a["mu_cth"] > COS70)
            & (a["pi_p"] > 150) & (a["pi_p"] < 1200) & (a["pi_cth"] > COS70))
    one = base & a["no_extra_pi"].astype(bool)
    sig = one & a["has_p"].astype(bool)
    return dict(pi_p=a["pi_p"], W=a["W"], w=w, base=base, one=one, sig=sig, prot=a["has_p"].astype(bool))


def cond(d, key, edges, num, den):
    out = []
    for i in range(len(edges) - 1):
        m = den & (d[key] >= edges[i]) & (d[key] < edges[i + 1])
        out.append(d["w"][m & num].sum() / max(d["w"][m].sum(), 1e-30))
    return np.array(out)


def dsig(d, key, edges, sel):
    h, _ = np.histogram(d[key][sel], edges, weights=d["w"][sel])
    return h / np.diff(edges)


def main():
    A, D = ach_masks(), ado_masks()
    Ped = np.array([150, 300, 450, 600, 750, 900, 1050, 1200.])
    Wed = np.array([1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000.])
    pc, wc = 0.5 * (Ped[1:] + Ped[:-1]), 0.5 * (Wed[1:] + Wed[:-1])

    print("=== P(in-window proton | pi_p)  (signal base, pre-proton-cut) ===")
    aP, dP = cond(A, "pi_p", Ped, A["prot"], A["one"]), cond(D, "pi_p", Ped, D["prot"], D["one"])
    print(f"{'pi_p':>8} {'ACH':>7} {'ADO':>7} {'ADO/ACH':>8}")
    for i in range(len(pc)):
        print(f"{pc[i]:8.0f} {aP[i]:7.3f} {dP[i]:7.3f} {dP[i]/max(aP[i],1e-9):8.3f}")
    print("\n=== P(in-window proton | W) ===")
    aW, dW = cond(A, "W", Wed, A["prot"], A["one"]), cond(D, "W", Wed, D["prot"], D["one"])
    print(f"{'W':>8} {'ACH':>7} {'ADO':>7} {'ADO/ACH':>8}")
    for i in range(len(wc)):
        print(f"{wc[i]:8.0f} {aW[i]:7.3f} {dW[i]:7.3f} {dW[i]/max(aW[i],1e-9):8.3f}")

    print("\n=== cut-ladder ACH/ADO ratio of dsigma/dpi_p (per stage) ===")
    for st in ("base", "one", "sig"):
        a, d = dsig(A, "pi_p", Ped, A[st]), dsig(D, "pi_p", Ped, D[st])
        r = a / np.where(d > 0, d, np.nan)
        print(f"  {st:5s} ACH/ADO:", "  ".join(f"{x:5.2f}" for x in r),
              f"| sig ACH={A['w'][A[st]].sum():.3e} ADO={D['w'][D[st]].sum():.3e}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(pc, aP, "o-", color="0.35", label="ACHILLES"); ax[0].plot(pc, dP, "s-", color="C0", label="ADoNIS")
    ax[0].set_xlabel(r"$p_\pi$ [MeV/c]"); ax[0].set_ylabel("P(in-window proton)"); ax[0].set_title("Proton-leg acceptance vs pion momentum"); ax[0].legend(); ax[0].set_ylim(0, 0.7)
    ax[1].plot(wc, aW, "o-", color="0.35", label="ACHILLES"); ax[1].plot(wc, dW, "s-", color="C0", label="ADoNIS")
    ax[1].set_xlabel("vertex W [MeV]"); ax[1].set_ylabel("P(in-window proton)"); ax[1].set_title("Proton-leg acceptance vs W"); ax[1].legend(); ax[1].set_ylim(0, 0.7)
    for st, col, lab in (("base", "0.6", "base (pre-proton)"), ("sig", "C3", "full signal")):
        a, d = dsig(A, "pi_p", Ped, A[st]), dsig(D, "pi_p", Ped, D[st])
        ax[2].plot(pc, a / np.where(d > 0, d, np.nan), "o-" if st == "sig" else "x--", color=col, label=f"ACH/ADO {lab}")
    ax[2].axhspan(0.9, 1.1, color="green", alpha=0.12); ax[2].axhline(1.0, ls=":", color="green")
    ax[2].set_xlabel(r"$p_\pi$ [MeV/c]"); ax[2].set_ylabel("ACH/ADO d$\\sigma$/d$p_\\pi$"); ax[2].set_title("Tail ratio: base vs full signal"); ax[2].legend(); ax[2].set_ylim(0.4, 1.4)
    fig.tight_layout(); fig.savefig("paper_figures/cc1pi_protonleg_bisect.png", dpi=120)
    print("\nwrote paper_figures/cc1pi_protonleg_bisect.png")


if __name__ == "__main__":
    main()
