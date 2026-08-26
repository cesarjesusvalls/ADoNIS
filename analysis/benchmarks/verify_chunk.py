"""Does chunking the EVENT axis reproduce the unchunked weights and binned model exactly, and what does
it cost in wall time?

FSI records are ragged (per interaction slot, with an event index), so a chunk window must select its
slots, rebase their indices, and pad to a common width -- an error there is silent (still a plausible
number, just wrong for some events).  Compares, on a real bank: per-event weights, the binned model
through BinSpec, and wall time per model evaluation, chunked vs unchunked.

Usage:  srun ... python -m analysis.benchmarks.verify_chunk [--sig-cap N] [--chunks 20000,50000,0]
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P2.yaml")
    ap.add_argument("--sig-cap", type=int, default=25000)
    ap.add_argument("--chunks", default="5000,12500,25000")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    import dataclasses, jax, jax.numpy as jnp
    from adonis.fit.config import FitConfig
    from adonis.fit.kernels import bank_windows, bank_weight_window
    from analysis.campaign.stages import multisample as MS
    from adonis.reweight import bank_reweight as BR
    from adonis.reweight.reweight_model import nominal_knobs
    from adonis.reweight.knobs import knobs_of, theta_nominal
    from adonis.fit import binning as IC

    log(f"jax {jax.__version__} devices={jax.devices()}")
    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    eng = MS.build_multisample_engine(log, cfg)
    s = eng.samples[0]
    log(f"sample {s.name}: {s.n_events:,} events, {len(s.ds)} datasets")

    nom = s.nom
    th = np.asarray(theta_nominal(nom), float); th[0] *= 1.07; th[1] *= 0.93
    kn = knobs_of(jnp.asarray(th), nom)
    w_ref = np.asarray(s._wf(jnp.asarray(th), s.JB))
    m_ref = np.concatenate([IC.bin_w(d, w_ref) for d in s.ds])
    log(f"reference: |w-1| max {np.abs(w_ref-1).max():.3e}, model {m_ref.shape}")

    rows = []
    for C in [int(x) for x in a.chunks.split(",") if x.strip()]:
        t_b = time.perf_counter()
        nch, Cc, subs, owns = bank_windows(s.JB, s.n_events, C)
        t_build = time.perf_counter() - t_b
        f = jax.jit(lambda B: bank_weight_window(BR, B, kn, s.grids, Cc))
        w_chunk = np.zeros(s.n_events); seen = np.zeros(s.n_events, bool)
        for wi in range(nch):
            wv = np.asarray(f(subs[wi]))
            lo, hi = owns[wi]
            st = wi * Cc if wi < nch - 1 else s.n_events - Cc
            w_chunk[lo:hi] = wv[lo - st:hi - st]; seen[lo:hi] = True
        assert seen.all(), "windows did not cover every event"
        d_w = float(np.max(np.abs(w_chunk - w_ref)) / max(np.max(np.abs(w_ref)), 1e-300))
        m_chunk = np.concatenate([IC.bin_w(d, w_chunk) for d in s.ds])
        d_m = float(np.max(np.abs(m_chunk - m_ref)) / max(np.max(np.abs(m_ref)), 1e-300))
        def run_chunked():
            for wi in range(nch):
                np.asarray(f(subs[wi]))
        run_chunked()
        ts = []
        for _ in range(a.reps):
            t1 = time.perf_counter(); run_chunked(); ts.append(time.perf_counter() - t1)
        rows.append((C, nch, d_w, d_m, float(np.median(ts)), t_build))
        log(f"  chunk {C:>7,} -> {nch:2d} windows | w max rel {d_w:.2e} | model max rel {d_m:.2e} "
            f"| {1e3*np.median(ts):7.1f} ms | build {t_build:.1f}s")

    np.asarray(s._wf(jnp.asarray(th), s.JB))
    tu = []
    for _ in range(a.reps):
        t1 = time.perf_counter(); np.asarray(s._wf(jnp.asarray(th), s.JB)); tu.append(time.perf_counter()-t1)
    t_un = float(np.median(tu))
    print(f"\n==== event chunking on {s.name} ({s.n_events:,} events) ====")
    print(f"{'chunk':>9} {'windows':>8} {'w max rel':>11} {'model max rel':>14} {'ms':>9} "
          f"{'vs unchunked':>13} {'build s':>9}")
    print(f"{'none':>9} {1:>8} {0.0:11.2e} {0.0:14.2e} {1e3*t_un:9.1f} {1.0:12.2f}x {0.0:9.1f}")
    for C, nch, dw, dm, t, tb in rows:
        print(f"{C:>9,} {nch:>8} {dw:11.2e} {dm:14.2e} {1e3*t:9.1f} {t/t_un:12.2f}x {tb:9.1f}")


if __name__ == "__main__":
    sys.exit(main())
