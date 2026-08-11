"""Feldman-Cousins confidence belt for one dial (the Neyman construction with LR ordering).

Why: the quoted sigma comes from the Gauss-Newton curvature at the best fit, which assumes chi2 is
quadratic and theta interior.  Neither holds for every dial here -- E_b sits 1.2 sigma from a hard wall,
and M_A_res/S_Delta lie along a curved 11:1 valley whose profile departs from the parabola by more than
the parabola's own height.  Measured against 500 fixed-truth refits, the Gaussian interval covers 53.6%
(not 68.3%) for E_b.  FC fixes this BY CONSTRUCTION: coverage is exact whatever the shape, because the
critical value is read off the actual sampling distribution of the test statistic at each candidate truth,
rather than assumed to be 1.0.

  for each candidate true value t on a grid:
      throw M pseudo-experiments with the POI at t (other dials at their reference values)
      for each: chi2_num = min over the other 15 dials with POI FIXED at t
                chi2_den = min over all 16 (the global fit)
                dchi2 = chi2_num - chi2_den          >= 0 by construction
      crit[t] = the 68th / 90th percentile of that dchi2 sample     <- NOT 1.00 / 2.71 in general
  the interval from the observed data is  { t : dchi2_obs(t) < crit[t] }

The denominator fit is the expensive half, so both fits warm-start from the throw truth.  Shard over a
SLURM array with S4_FC_BASE / S4_FC_NGRID; one npz per shard, merged by the figure.

Env: S4_FC_DIAL (knob name, required), S4_FC_LO/S4_FC_HI/S4_FC_NGRID (grid, defaults per dial below),
     S4_FC_NTOY (toys per grid point, default 200), S4_FC_BASE (first grid index this shard does),
     S4_FC_NPT (grid points this shard does), S4_FIXED_TRUTH (the reference point), S4_LOGFIT,
     ALTGEN_NIT + the S4_*_CHUNKS caps.  Writes output/altgen/<label>_fc_<dial>_<base>.npz
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine, MULTISAMPLE_NPZ
from analysis.paper.physfit.physical_fit_run import lm_fit, parse_inject
from adonis.reweight.reweight_model import nominal_knobs

# Grids span roughly +-3.5 quoted sigma about the reference, clipped to the physical region.  E_b runs
# down to the wall so the belt can show the two-sided -> upper-limit transition FC is built to handle.
DEFAULT_GRID = {"Eb_shift": (0.02, 1.6), "M_A_res": (0.55, 1.20), "delta_strength": (0.45, 1.65)}


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    DIAL = os.environ["S4_FC_DIAL"]
    NTOY = int(os.environ.get("S4_FC_NTOY", "200"))
    NGRID = int(os.environ.get("S4_FC_NGRID", "13"))
    BASE = int(os.environ.get("S4_FC_BASE", "0"))
    NPT = int(os.environ.get("S4_FC_NPT", str(NGRID)))
    NIT = int(os.environ.get("ALTGEN_NIT", "20"))
    LABEL = os.environ.get("ADONIS_LABEL", "sec4_ref")
    FIXED = os.environ["S4_FIXED_TRUTH"]
    SEEDOFF = int(os.environ.get("S4_FC_SEEDOFF", "0"))   # toy-shard index: disjoint draws

    eng = build_multisample_engine(log)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    real_prior = eng.prior.copy()
    eng.prior = eng.prior * 1e6                      # MLE, matching the closure and the ensemble
    ref, _ = parse_inject(FIXED, nominal_knobs())
    k = eng.pnames.index(DIAL); c = subset.index(k)
    others = [j for j in subset if j != k]

    lo, hi = DEFAULT_GRID.get(DIAL, (0.5 * ref[k], 1.5 * ref[k]))
    lo = float(os.environ.get("S4_FC_LO", lo)); hi = float(os.environ.get("S4_FC_HI", hi))
    grid = np.linspace(lo, hi, NGRID)
    mine = list(range(BASE, min(BASE + NPT, NGRID)))
    log(f"FC belt for {DIAL}: reference={ref[k]:.4f}, grid [{lo:g},{hi:g}] x{NGRID}, "
        f"{NTOY} toys/point; this shard does points {mine}")

    dchi2 = np.full((NGRID, NTOY), np.nan)
    fit_poi = np.full((NGRID, NTOY), np.nan)
    out = f"output/altgen/{LABEL}_fc_{DIAL.replace('[', '').replace(']', '')}_{BASE:02d}_{SEEDOFF}.npz"

    def save(final=False):
        np.savez(out, dial=DIAL, grid=grid, my_points=np.array(mine, int), dchi2=dchi2, fit_poi=fit_poi,
                 ref=ref, subset=subset, pnames=eng.pnames, ntoy=NTOY, complete=bool(final),
                 prior=real_prior)
        if final:
            log(f"[out] {out}")

    # ASIMOV mode: no toys, just dchi2_obs(t) on the NOISELESS data at the reference point.  This is the
    # observed curve the belt is read against -- the FC interval is {t : dchi2_obs(t) < crit(t)} -- and it
    # has to live on the SAME grid as the belt, which the sigma-spaced profile scan does not.
    if os.environ.get("S4_FC_ASIMOV", "") == "1":
        eng.set_closure_data(ref)                              # noiseless data AT the reference truth
        obs = np.full(NGRID, np.nan)
        _, _, _, _, _, cd_glob = lm_fit(eng, subset, "fc_asimov_glob", nit=NIT, th_init=ref.copy())
        for gi in range(NGRID):
            t_fix = ref.copy(); t_fix[k] = grid[gi]
            _, _, _, _, _, cd = lm_fit(eng, others, f"fc_asimov{gi}", nit=NIT, th_init=t_fix)
            obs[gi] = cd - cd_glob
            log(f"  asimov point {gi} ({DIAL}={grid[gi]:.4f}): dchi2_obs={obs[gi]:.4f}")
        oo = f"output/altgen/{LABEL}_fcasimov_{DIAL.replace('[', '').replace(']', '')}.npz"
        np.savez(oo, dial=DIAL, grid=grid, dchi2_obs=obs, chi2_global=cd_glob, ref=ref,
                 subset=subset, pnames=eng.pnames)
        log(f"[out] {oo}")
        return

    for gi in mine:
        # PROFILE CONSTRUCTION.  The nuisances need true values for the throw too, and strict FC would
        # scan all 15 of them -- infeasible.  The standard substitute (MINOS/NOvA) is to throw them at
        # their CONDITIONAL best fit given theta = t on the OBSERVED data, which is what this does.  The
        # earlier version threw them at the reference truth; at an Asimov point those coincide only at
        # t = truth and drift apart as t moves, since the other dials shift to absorb the pinned one.
        eng.set_closure_data(ref)                              # the observed (noiseless) dataset
        th_c = ref.copy(); th_c[k] = grid[gi]
        th_cond, *_ = lm_fit(eng, others, f"cond[{gi}]", nit=NIT, th_init=th_c)
        t_true = th_cond.copy(); t_true[k] = grid[gi]          # (t, nu-hat-hat(t))
        for it in range(NTOY):
            rng = np.random.default_rng(7_000_000 + 10_000 * gi + SEEDOFF * 1000 + it)
            eng.set_closure_data(t_true)                       # noiseless prediction at this candidate
            for d in eng.ds:                                   # + per-bin stat throw at the fit's sigma
                sd = np.where(np.isfinite(d["sigma"]), d["sigma"], 0.0)
                d["data"] = d["data"] + rng.normal(0.0, sd)
            # numerator: POI FIXED at the candidate, the other 15 free
            th_n, *_, cd_num = lm_fit(eng, others, f"fc{gi}.{it}n", nit=NIT, th_init=t_true.copy())
            # denominator: the global 16-dial fit.  Warm-started from the numerator solution, which is
            # already minimised in 15 of the 16 directions -- a few LM steps instead of a cold walk.
            th_d, *_, cd_den = lm_fit(eng, subset, f"fc{gi}.{it}d", nit=NIT, th_init=th_n.copy())
            dchi2[gi, it] = max(cd_num - cd_den, 0.0)          # >= 0 up to optimiser noise
            fit_poi[gi, it] = th_d[k]
        q68, q90 = np.percentile(dchi2[gi], [68.27, 90.0])
        log(f"  point {gi} ({DIAL}={grid[gi]:.4f}): crit68={q68:.3f} crit90={q90:.3f} "
            f"(Wilks would say 1.000 / 2.706)")
        save()

    save(final=True)


if __name__ == "__main__":
    main()
