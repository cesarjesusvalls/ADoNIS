"""BANK the ADoNIS CC1pi pre-selection per-event quantities from the CENTRAL res_C chain
(cc1pi_fig_tki.res_C return_raw=True: primary RES + pion cascade + nucleon cascade + knockouts,
exactly the figure path). This is the ADoNIS analog of the ACHILLES res_w_FSI extraction
(scripts/extract_t2k_res_w.py) so the signal cut ladder can be applied IDENTICALLY off-line --
the substitution-bisect of the proton leg vs pion momentum. One generation -> all analysis free.

Per event (carbon, t-channel sampler, spline amps2): mu_p/mu_cth, post-FSI pi_p/pi_cth/pid_pi,
leading in-window proton lp_p/lp_cth + has_p (== ACHILLES prot_ok), no_extra_pi (== n_pi==1 proxy),
vertex W/Q2, the primary RES pion charge ppid0, and the absolute weight w.

Usage: python -u scripts/gen_cc1pi_raw.py [NRES=150000] [NSEED=4]
Output: data/oracle/t2k_cc1pi_raw_adonis.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scripts.cc1pi_fig_tki as F        # pins spline; res_C is the central chain

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def main():
    parts = []
    for sd in range(NSEED):
        d = F.res_C(NRES, sd, return_raw=True)
        d["w"] = d["w"] / NSEED
        parts.append(d)
        print(f"  seed {sd+1}/{NSEED}: +{len(d['w'])} events  sig={d['w'].sum()*NSEED:.4e} nb", flush=True)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez("data/oracle/t2k_cc1pi_raw_adonis.npz", **out)
    n = len(out["w"])
    sig = ((out["pid_pi"] == 211) & out["has_p"] & out["no_extra_pi"]
           & (out["mu_p"] > 250) & (out["mu_p"] < 7000) & (out["mu_cth"] > F.COS70)
           & (out["pi_p"] > 150) & (out["pi_p"] < 1200) & (out["pi_cth"] > F.COS70))
    print(f"banked {n} primary events  selected signal sig={out['w'][sig].sum():.4e} nb", flush=True)
    print("wrote data/oracle/t2k_cc1pi_raw_adonis.npz", flush=True)


if __name__ == "__main__":
    main()
