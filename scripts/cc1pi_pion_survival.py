"""Pion FSI survival/redistribution vs momentum, ADoNIS (from the banked pion fate) vs ACHILLES
(res_w FSI/no-FSI oracles).  All re-binning of cached files -- no cascade run.
  - survival(pi_p) = dsigma/dpi_p(FSI pi+) / dsigma/dpi_p(no-FSI pi+), carbon, mu+pi forward acceptance
    (same definition both generators; ADoNIS FSI=post-cascade, no-FSI=pre-cascade, SAME events).
  - redistribution: of primary pi+ with pi_p_pre > 1200 (above the signal window), what fraction
    down-scatter to a surviving pi+ inside [150,1200]?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.data.oracle.normalization import hepmc_norm
COS70 = np.cos(np.deg2rad(70.0))
ED = np.linspace(150, 1400, 14); CTR = 0.5 * (ED[1:] + ED[:-1]); BW = np.diff(ED)


def ach_survival():
    def load(f, h):
        a = np.load(f"data/oracle/{f}.npz"); w2 = hepmc_norm(h)["weight_to_nb"]
        m = ((np.asarray(a["pi_pid"]) == 211) & (np.asarray(a["pstr"]) > 1.0)
             & (np.asarray(a["mu_p"]) > 250) & (np.asarray(a["mu_cth"]) > COS70) & (np.asarray(a["pi_cth"]) > COS70))
        return np.asarray(a["pi_p"])[m], np.asarray(a["w"])[m] * w2
    fp, fw = load("t2k_res_w_achilles_FSI", "_oracle_out/T2K_CH_virt.hepmc")
    np_, nw = load("t2k_res_w_achilles_nofsi", "_oracle_out/T2K_CH_virt_nofsi.hepmc")
    hf, _ = np.histogram(fp, ED, weights=fw); hn, _ = np.histogram(np_, ED, weights=nw)
    return np.where(hn > 0, hf / hn, np.nan)


def ado_survival():
    d = np.load("data/oracle/t2k_cc1pi_pion_fate.npz")
    kmu = np.asarray(d["k_mu"]); w = np.asarray(d["w"])
    pre, post = np.asarray(d["pi_pre"]), np.asarray(d["pi_post"])
    mu_m = (np.linalg.norm(kmu[:, 1:], axis=1) > 250) & (kmu[:, 3] / np.linalg.norm(kmu[:, 1:], axis=1) > COS70)
    pp_pre = np.linalg.norm(pre[:, 1:], axis=1); cth_pre = pre[:, 3] / np.clip(pp_pre, 1e-9, None)
    pp_post = np.linalg.norm(post[:, 1:], axis=1); cth_post = post[:, 3] / np.clip(pp_post, 1e-9, None)
    nofsi = (np.asarray(d["pid_pre"]) == 211) & (cth_pre > COS70) & mu_m
    fsi = (np.asarray(d["pid_post"]) == 211) & (cth_post > COS70) & mu_m
    hn, _ = np.histogram(pp_pre[nofsi], ED, weights=w[nofsi])
    hf, _ = np.histogram(pp_post[fsi], ED, weights=w[fsi])
    surv = np.where(hn > 0, hf / hn, np.nan)
    # redistribution: primary pi+ above the window (pi_p_pre>1200) that survive as pi+ inside [150,1200]
    hi = (np.asarray(d["pid_pre"]) == 211) & (pp_pre > 1200) & mu_m
    down = hi & (np.asarray(d["pid_post"]) == 211) & (pp_post >= 150) & (pp_post <= 1200) & (cth_post > COS70)
    frac_down = w[down].sum() / max(w[hi].sum(), 1e-30)
    return surv, frac_down


def main():
    asv = ach_survival(); dsv, fdown = ado_survival()
    print(f"{'pi_p[MeV]':>9} {'ACH surv':>9} {'ADO surv':>9} {'ADO/ACH':>9}")
    for i in range(len(CTR)):
        if np.isfinite(asv[i]) and np.isfinite(dsv[i]):
            print(f"{CTR[i]:9.0f} {asv[i]:9.3f} {dsv[i]:9.3f} {dsv[i]/asv[i]:9.3f}")
    print(f"\nADoNIS: primary pi+ with pi_p>1200 that down-scatter into [150,1200] as pi+: {fdown:.3f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(CTR, asv, "o-", color="0.35", label="ACHILLES (FSI/noFSI)")
    ax.plot(CTR, dsv, "s-", color="C0", label="ADoNIS (post/pre)")
    ax.axhline(1.0, ls=":", color="0.6"); ax.set_ylim(0, 1.3)
    ax.set_xlabel(r"$p_\pi$ [MeV/c]"); ax.set_ylabel(r"$\pi^+$ survival in window (FSI/no-FSI)")
    ax.set_title("Carbon $\\pi^+$ FSI survival vs momentum: ADoNIS vs ACHILLES"); ax.legend()
    fig.tight_layout(); fig.savefig("paper_figures/cc1pi_pion_survival.png", dpi=120)
    print("wrote paper_figures/cc1pi_pion_survival.png")


if __name__ == "__main__":
    main()
