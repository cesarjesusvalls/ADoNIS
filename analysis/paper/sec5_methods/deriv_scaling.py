"""Sec 5 -- how the autodiff derivative cost scales with ORDER, measured vs theory.

An order-k non-Gaussian corner needs the model's k-th derivative tensor, i.e. every mixed k-th directional
derivative over the nsub dials.  Two things scale with k:
  * the NUMBER of independent tensor entries      N_k = C(nsub + k - 1, k)   (combinations w/ replacement),
  * the COST of ONE entry (a depth-k nested jvp)  t_term(k) ~ t1 * rho^(k-1) (each forward-over-forward
    layer re-differentiates the whole graph, so work multiplies by a ~constant factor rho per order).
So the full order-k build is  T_k = N_k * t_term(k) ~ C(nsub+k-1,k) * rho^(k-1)  -- combinatorial x exponential.

This script MEASURES both factors on the real multisample engine: t_term(k) from a few timed depth-k calls
(jit-compile time recorded separately), and -- for the low orders where it is cheap -- the ACTUAL full-tensor
build time T_k, to check T_k ~= N_k * t_term(k).  Predicted T_k for the higher orders (too expensive to build
in full) then follow from the measured per-term cost x the exact combinatorial count.

Env: S4_*_CHUNKS bank caps (as for the closure); DERIV_KMAX (max order to time, default 6);
DERIV_BUILD_MAX (max order to build IN FULL for validation, default 3); DERIV_REPS (timed reps, default 5).
Writes output/altgen/deriv_scaling.npz.
"""
import os
import sys
import math
import time
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine
from analysis.paper.physical_fit import NPAR


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    KMAX = int(os.environ.get("DERIV_KMAX", "6"))
    BUILD_MAX = int(os.environ.get("DERIV_BUILD_MAX", "3"))
    REPS = int(os.environ.get("DERIV_REPS", "5"))

    LABEL = os.environ.get("ADONIS_LABEL", "sec4_closure_r16_noprior")
    z = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; truth = np.asarray(z["truth"]); nsub = len(sub)
    log(f"nsub={nsub}  KMAX={KMAX}  BUILD_MAX={BUILD_MAX}  REPS={REPS}")

    eng = build_multisample_engine(log)
    eng.set_closure_data(truth)
    bfp = truth.copy()

    def evec(i):
        v = np.zeros(NPAR); v[sub[i]] = 1.0; return v

    # --- self-check: generic ddn must reproduce the hand-written dd (order 2) and dd3 (order 3) --------- #
    e0, e1, e2 = evec(0), evec(1), evec(2)
    a2 = eng.ddn(bfp, [e0, e0]); b2 = eng.dd(bfp, e0, 2)
    a3 = eng.ddn(bfp, [e0, e1, e2]); b3 = eng.dd3(bfp, e0, e1, e2)
    r2 = np.max(np.abs(a2 - b2)); r3 = np.max(np.abs(a3 - b3))
    log(f"self-check: |ddn2-dd2|={r2:.2e}  |ddn3-dd3|={r3:.2e}")
    assert r2 < 1e-8 and r3 < 1e-8, "generic ddn disagrees with hand-written dd/dd3"

    out = "output/altgen/deriv_scaling.npz"
    def save(rows):
        np.savez(out, nsub=nsub, order=np.array([r[0] for r in rows]),
                 n_terms=np.array([r[1] for r in rows], dtype=float),
                 t_term=np.array([r[2] for r in rows]), t_term_std=np.array([r[3] for r in rows]),
                 t_compile=np.array([r[4] for r in rows]), pred_full=np.array([r[5] for r in rows]),
                 t_full=np.array([r[6] for r in rows]))

    rows = []
    for k in range(1, KMAX + 1):
        try:
            n_terms = math.comb(nsub + k - 1, k)                 # C(nsub+k-1, k) unique symmetric entries
            tang = [evec(i % nsub) for i in range(k)]            # a representative mixed k-term
            tc = time.perf_counter(); _ = eng.ddn(bfp, tang); t_compile = time.perf_counter() - tc  # incl. jit
            ts = []
            for _r in range(REPS):
                tc = time.perf_counter(); _ = eng.ddn(bfp, tang); ts.append(time.perf_counter() - tc)
            t_term = float(np.median(ts))
            pred_full = n_terms * t_term
            # actual full-tensor build (only for the cheap low orders) -> a measured T_k to validate the model
            t_full = np.nan
            if k <= BUILD_MAX:
                tb = time.perf_counter()
                for combo in itertools.combinations_with_replacement(range(nsub), k):
                    _ = eng.ddn(bfp, [evec(i) for i in combo])
                t_full = time.perf_counter() - tb
            rows.append((k, n_terms, t_term, float(np.std(ts)), t_compile, pred_full, t_full))
            log(f"k={k}: N={n_terms:>7d}  t_term={t_term*1e3:8.1f} ms  compile={t_compile:6.2f}s  "
                f"pred_full={pred_full:9.1f}s" + (f"  MEASURED_full={t_full:8.1f}s" if k <= BUILD_MAX else ""))
            save(rows)                                            # checkpoint after every order
        except Exception as e:                                   # OOM / compile blowup at high k -> keep what we have
            log(f"k={k}: FAILED ({type(e).__name__}: {e}) -- stopping, lower orders saved")
            break
    log(f"[out] {out}  ({len(rows)} orders)")


if __name__ == "__main__":
    main()
