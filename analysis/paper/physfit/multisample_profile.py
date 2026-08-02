"""Section 4 profile study: is the fit's QUADRATIC (Gauss-Newton, at-BFP) error actually the likelihood?

The closure quotes sigma = sqrt(diag(V)), V = (J^T W J + prior)^-1 at the BFP -- the parabolic (Laplace)
approximation of the likelihood curvature.  This checks it directly, per dial, by PROFILING the EXACT
nonlinear chi2:

  for each fitted dial k:
    * scan theta_k over BFP +/- 3 sigma_post,k  (marginal sigma from V),
    * at each grid node FIX theta_k and re-minimize chi2 over the OTHER 15 dials (LM) -> profile chi2(theta_k),
    * compare Delta chi2_profile(theta_k)  vs  the parabola (theta_k - BFP_k)^2 / sigma_post,k^2.

If the two agree over +/-3 sigma the marginal error is trustworthy; a dial whose exact profile is skewed or
walls off (Eb_shift's floor is the candidate) shows up as a departure.  Reuses the closure's engine + fit;
reads the BFP + injected data from an existing closure run (so it profiles the SAME likelihood).

Env: ADONIS_LABEL (closure npz to read + profile, default sec4_closure_random16),
     S4_PROFILE_N (grid nodes per side, default 6), ALTGEN_NIT (inner-fit iters, default 20)
     + the S4_*_CHUNKS bank caps.  Writes output/altgen/<label>_profile.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine
from analysis.paper.physfit.physical_fit_run import lm_fit
from analysis.paper.physical_fit import PNAMES


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    LABEL = os.environ.get("ADONIS_LABEL", "sec4_closure_random16")
    NG = int(os.environ.get("S4_PROFILE_N", "6"))          # nodes per side (2*NG+1 total)
    NIT = int(os.environ.get("ALTGEN_NIT", "20"))

    z = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    subset = [int(k) for k in z["subset"]]
    bfp = np.asarray(z["fit_th"] if "fit_th" in z.files else z["M0_th"])
    V = np.asarray(z["fit_V"] if "fit_V" in z.files else z["M0_V"])
    spost = np.sqrt(np.abs(np.diag(V)))                    # marginal sigma per dial (subset order)
    truth = np.asarray(z["truth"])
    log(f"profiling {LABEL}: {len(subset)} dials, BFP loaded")

    eng = build_multisample_engine(log)
    nom0 = eng.th0.copy()                                  # true nominal = the prior centre (never mutated)
    eng.set_closure_data(truth)                            # reproduce the closure's (noiseless) data
    data = np.concatenate([d["data"] for d in eng.ds]); sigma = np.concatenate([d["sigma"] for d in eng.ds])
    PRIOR_SCALE = float(os.environ.get("S4_PRIOR_SCALE", "1.0"))   # 0 -> MLE profile (data-only, no prior)
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
        log(f"FIT/objective prior width scaled x{PRIOR_SCALE} (MLE profile)")

    def objective(th):
        """FULL negative-log-posterior = chi2_data + prior penalty (the curvature sigma_post encodes)."""
        cd = float(np.sum(((eng.model(th) - data) / sigma)**2))
        pri = float(np.sum([(th[j] - nom0[j])**2 / eng.prior[j]**2 for j in subset]))
        return cd + pri, cd

    L_min, cd_min = objective(bfp)
    log(f"objective at BFP: total={L_min:.2f}  chi2_data={cd_min:.2f}")

    grid = np.linspace(-3.0, 3.0, 2 * NG + 1)              # theta_k - BFP_k in units of sigma_post,k
    prof = np.full((len(subset), len(grid)), np.nan)       # EXACT profiled Delta(total objective)
    profd = np.full((len(subset), len(grid)), np.nan)      # (also keep Delta chi2_data)
    for c, k in enumerate(subset):
        free = [j for j in subset if j != k]               # re-minimise over the OTHER 15 dials
        for gi, gv in enumerate(grid):
            val = max(bfp[k] + gv * spost[c], 1e-2) if eng.pnames[k] == "Eb_shift" else bfp[k] + gv * spost[c]
            th_init = bfp.copy(); th_init[k] = val          # warm-start from BFP with dial k pinned
            th, _V, _J, _m, _c, _cd = lm_fit(eng, free, f"prof[{eng.pnames[k]}]{gi}", nit=NIT, th_init=th_init)
            th[k] = val                                     # ensure the pinned value (lm_fit never moves it)
            Lval, cdval = objective(th)
            prof[c, gi] = Lval - L_min; profd[c, gi] = cdval - cd_min
        log(f"  [{eng.pnames[k]}] profiled ({2*NG+1} nodes)  Dobj@+-3sig="
            f"[{prof[c,0]:.1f},{prof[c,-1]:.1f}] (parabola 9.0)")

    out = f"output/altgen/{LABEL}_profile.npz"
    np.savez(out, subset=subset, pnames=eng.pnames, grid_sigma=grid, prof_dobj=prof, prof_dchi2=profd,
             bfp=bfp, sigma_post=spost, truth=truth, chi2_min=cd_min)
    log(f"[out] {out}")


if __name__ == "__main__":
    main()
