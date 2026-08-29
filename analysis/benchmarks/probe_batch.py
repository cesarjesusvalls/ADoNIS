"""Largest Jacobian dial-batch that fits on this device, at a given event count.

The workload grid must use ONE batch B at every N: with batch B a jacobian costs ceil(n/B) primal passes
instead of 1, so letting B fall as N rises would make Gauss-Newton look more expensive at large N for a
device-memory reason, not an algorithmic one.  So the largest N sets B for the entire grid.

Batches are tried UPWARD, since a device OOM poisons the CUDA context and anything attempted after a
failure is unreliable.

    srun ... python -m analysis.benchmarks.probe_batch --sig-cap 250000 [--ndials 17]
"""
from __future__ import annotations

from analysis._cli import results_dir, FLAT_PRIOR_SCALE, timed_log
import argparse, dataclasses, gc, sys, time
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P1.yaml")
    ap.add_argument("--sig-cap", type=int, required=True)
    ap.add_argument("--ndials", type=int, default=17)
    ap.add_argument("--batches", default="1,2,3,4,6,8,12,17")
    ap.add_argument("--stages", default="chi2,residuals,jac,grad",
                    help="which programs to build, in increasing order of memory")
    a = ap.parse_args(argv)

    log = timed_log()

    import jax
    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from adonis.fit.kernels import FitKernel
    from analysis.benchmarks._shared import _dial_order, freeze_sample
    from analysis.campaign.stages import multisample as MS
    from adonis.reweight.reweight_model import nominal_knobs

    log(f"jax {jax.__version__} devices={jax.devices()} x64={jax.config.jax_enable_x64}")
    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    eng = MS.build_multisample_engine(log, cfg)
    g = np.load(MS.MULTISAMPLE_NPZ, allow_pickle=True)
    subset = sorted(_dial_order(g, eng.pnames)[:a.ndials])
    freeze_sample(eng, np.load(str(results_dir() / "sec4_P1.npz"), allow_pickle=True), log)
    truth, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    th0 = np.asarray(eng.th0, float)
    t = th0.copy(); t[np.asarray(subset, int)] = truth[np.asarray(subset, int)]
    eng.set_closure_data(t)
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)

    def mem():
        try:
            m = jax.local_devices()[0].memory_stats()
            return (f"{m['bytes_in_use']/2**30:.2f}/{m['bytes_limit']/2**30:.2f} GB in use, "
                    f"peak {m['peak_bytes_in_use']/2**30:.2f} GB")
        except Exception:
            return "memory_stats unavailable"

    stages = [s_.strip() for s_ in a.stages.split(",") if s_.strip()]
    ok, k, rows = [], None, []
    log(f"device before anything: {mem()}")
    for B in [int(s_) for s_ in a.batches.split(",")]:
        if B > a.ndials:
            break
        if k is not None:
            del k; k = None
        gc.collect(); jax.clear_caches()
        k = FitKernel(eng, subset, jac_batch=B)
        log(f"  --- batch {B} --- kernel built (bin caches warm): {mem()}")
        for st in stages:
            try:
                t1 = time.perf_counter()
                if st == "chi2":
                    k.chi2(k.x0)
                elif st == "residuals":
                    k.residuals(k.x0)
                elif st == "jac":
                    k.jac_data(k.x0)
                elif st == "grad":
                    k.grad(k.x0)
                else:
                    raise SystemExit(f"unknown stage {st}")
                dt = time.perf_counter() - t1
                rows.append((B, st, True))
                log(f"  batch {B:2d} {st:>9}: OK   compile+run {dt:6.1f}s   {mem()}")
                if st == "jac":
                    ok.append(B)
            except Exception as e:
                s_ = str(e)
                if "RESOURCE_EXHAUSTED" in s_ or "OUT_OF_MEMORY" in s_.upper():
                    rows.append((B, st, False))
                    log(f"  batch {B:2d} {st:>9}: OOM  {mem()}")
                    log(f"  stopping: the CUDA context is unreliable after an OOM")
                    B = None
                    break
                raise
        if B is None:
            break

    print(f"\n==== {a.sig_cap:,} events/sample ({10*a.sig_cap:,} resident), n={a.ndials} ====")
    for B, st, good in rows:
        print(f"  batch {B:2d}  {st:>9}  {'OK' if good else 'OOM'}")
    print(f"  jac feasible batches : {ok}")
    print(f"  MAX FEASIBLE JAC BATCH: {max(ok) if ok else 'NONE'}")


if __name__ == "__main__":
    sys.exit(main())
