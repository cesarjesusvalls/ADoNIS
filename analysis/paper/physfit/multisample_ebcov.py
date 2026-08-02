"""Sec 4.3b -- coverage at a physical boundary: Gaussian interval vs profile interval for E_b near its wall.

Fix a near-wall truth (E_b* ~ 0.15 MeV), throw the other dials from the prior and STAT noise on the data,
fit (MLE), and for E_b record BOTH interval constructions and whether each contains E_b*:
  * Gaussian  [th_hat - sigma, th_hat + sigma]   -- symmetric, can dip below the wall,
  * profile   Delta chi2 = 1 crossings           -- one-sided when the plateau at the wall < 1.
Over the ensemble the Gaussian 68% interval UNDER-covers (the wall biases th_hat and the symmetric interval
misses low), while the profile interval covers ~68%.  This is the "coverage when the posterior isn't
chi2-distributed" demonstration.

Env: S4_NTOYS (default 40)  EB_STAR (default 0.15)  ALTGEN_NIT (default 40) + S4_*_CHUNKS.  Always MLE
(no prior).  Writes output/altgen/sec4_ebcov.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine, MULTISAMPLE_NPZ
from analysis.paper.physfit.physical_fit_run import lm_fit
from analysis.paper.physical_fit import PNAMES


def _cross(x, y, side, x0):                                       # x where y crosses 1 on `side` of x0
    m = x < x0 if side < 0 else x > x0
    xs, ys = x[m], y[m] - 1.0
    s = np.where(np.diff(np.signbit(ys)))[0]
    if len(s) == 0:
        return np.nan
    i = s[-1] if side < 0 else s[0]
    return xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    NTOYS = int(os.environ.get("S4_NTOYS", "40"))
    EB_STAR = float(os.environ.get("EB_STAR", "0.15"))
    NIT = int(os.environ.get("ALTGEN_NIT", "40"))

    eng = build_multisample_engine(log)
    eng.prior = eng.prior * 1e6                                   # MLE (no prior) for the fit + profile
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    real_prior = (np.load(MULTISAMPLE_NPZ, allow_pickle=True)["prior"])
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    eb_k = PNAMES.index("Eb_shift"); free = [k for k in subset if k != eb_k]
    log(f"Eb-coverage: {NTOYS} toys, Eb*={EB_STAR} MeV (wall at 0)")

    rows = []
    for t in range(NTOYS):
        rng = np.random.default_rng(5000 + t)
        star = eng.th0.copy()
        for k in subset:                                         # others from prior, Eb at the near-wall truth
            star[k] = eng.th0[k] + real_prior[k] * rng.standard_normal()
        star[eb_k] = EB_STAR
        eng.set_closure_data(star)
        for d in eng.ds:
            sd = np.where(np.isfinite(d["sigma"]), d["sigma"], 0.0)
            d["data"] = d["data"] + rng.normal(0.0, sd)
        th, V, J, m, c, cd = lm_fit(eng, subset, f"toy{t}", nit=NIT)
        eb_hat = th[eb_k]; eb_sig = float(np.sqrt(abs(V[subset.index(eb_k), subset.index(eb_k)])))
        chi2_min = cd
        # profile Eb: scan, pin Eb, re-min the others; find Delta chi2 = 1 crossings (wall makes it flat <0)
        gr = np.linspace(min(-0.4, eb_hat - 3 * eb_sig), eb_hat + 3 * eb_sig, 13)
        dch = np.zeros(len(gr))
        for gi, v in enumerate(gr):
            ti = th.copy(); ti[eb_k] = v
            dch[gi] = lm_fit(eng, free, "p", nit=NIT, th_init=ti)[5] - chi2_min
        p_lo = _cross(gr, dch, -1, eb_hat); p_hi = _cross(gr, dch, +1, eb_hat)
        p_lo = 0.0 if np.isnan(p_lo) else max(p_lo, 0.0)          # wall: interval bounded below by 0
        gcov = (eb_hat - eb_sig) <= EB_STAR <= (eb_hat + eb_sig)
        pcov = p_lo <= EB_STAR <= (p_hi if not np.isnan(p_hi) else np.inf)
        rows.append((eb_hat, eb_sig, p_lo, p_hi, gcov, pcov))
        log(f"toy {t}: eb_hat={eb_hat:.3f} sig={eb_sig:.3f} gauss[{eb_hat-eb_sig:.2f},{eb_hat+eb_sig:.2f}] "
            f"prof[{p_lo:.2f},{p_hi:.2f}] gcov={gcov} pcov={pcov}")
    R = np.array(rows, float)
    np.savez("output/altgen/sec4_ebcov.npz", eb_star=EB_STAR, eb_hat=R[:, 0], eb_sig=R[:, 1],
             prof_lo=R[:, 2], prof_hi=R[:, 3], gauss_cover=R[:, 4], prof_cover=R[:, 5])
    log(f"[out] output/altgen/sec4_ebcov.npz | Gaussian cover {R[:,4].mean():.0%} | profile cover {R[:,5].mean():.0%}")


if __name__ == "__main__":
    main()
