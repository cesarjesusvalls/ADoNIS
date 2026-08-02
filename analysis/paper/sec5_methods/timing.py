"""Sec 5 timing: measure the per-operation cost of the differentiable engine (anchors the scaling figure).

Times, on the real multisample model (all 6 banks), the primitives that the fit and the non-Gaussian
methods are built from, and the finite-difference alternative a non-differentiable generator would need:
  t_model   one forward model eval           (also the cost of a corner grid NODE)
  t_jac     autodiff Jacobian (16 jvp)        (the fit gradient / Fisher, EXACT)
  t_fdjac   finite-difference Jacobian        (2*16 model evals, central diff -- noisy, non-diff engine)
  t_d2m     one 2nd directional derivative    (nested jvp; full d2m tensor = n(n+1)/2 of these)
Saves the medians + n_events so fig_scaling.py can draw cost-vs-N and cost-vs-order-m with real prefactors.

Env: S4_*_CHUNKS bank caps (sets n_events); ADONIS_TAG (label suffix).  Writes output/altgen/sec5_timing_<tag>.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine
from analysis.paper.physical_fit import NPAR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs


def _med(fn, rep=5):
    fn()                                                          # warm up (jit compile)
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    TAG = os.environ.get("ADONIS_TAG", "1ch")

    eng = build_multisample_engine(log)
    from analysis.paper.physfit.multisample import MULTISAMPLE_NPZ
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    th = theta_nominal(nominal_knobs())
    n_events = int(sum(s.weights(th).shape[0] for s in eng.samples))
    log(f"n_events={n_events:,}  n_dials={len(subset)}  n_bins={eng.row0[-1]}")

    def fd_jac():                                                 # central finite-difference Jacobian
        eps = 1e-3
        for k in subset:
            tp = th.copy(); tp[k] += eps; eng.model(tp)
            tm = th.copy(); tm[k] -= eps; eng.model(tm)

    v = np.zeros(NPAR); v[subset[0]] = 1.0
    t_model = _med(lambda: eng.model(th))
    t_jac = _med(lambda: eng.jac(th, subset))
    t_fdjac = _med(fd_jac, rep=3)
    t_d2m = _med(lambda: eng.dd(th, v, 2))
    log(f"t_model={t_model*1e3:.0f}ms  t_jac(16 jvp)={t_jac*1e3:.0f}ms  "
        f"t_fdjac(32 eval)={t_fdjac*1e3:.0f}ms  t_d2m(1 dir)={t_d2m*1e3:.0f}ms")

    out = f"output/altgen/sec5_timing_{TAG}.npz"
    np.savez(out, n_events=n_events, n_dials=len(subset), n_bins=int(eng.row0[-1]),
             t_model=t_model, t_jac=t_jac, t_fdjac=t_fdjac, t_d2m=t_d2m)
    log(f"[out] {out}")


if __name__ == "__main__":
    main()
