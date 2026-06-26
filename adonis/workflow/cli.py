"""ADoNIS generation CLI -- the single entry point for forward generation.

Run as a module (from the repo root):

  # one process (in-process; JIT compiles once for this run)
  python -u -m adonis.workflow.cli configs/gen_c_qe.yaml --n-seeds 5 --n-per-seed 25000 --n-w 2048

  # parallel production: K worker processes, each M seeds of N events in ONE process (JIT once per
  # worker, reused across its seeds), each writing its own checkpointed `..._batchWW.npz`.
  python -u -m adonis.workflow.cli configs/gen_c_qe.yaml \
      --workers 6 --seeds-per-worker 5 --n-per-seed 25000 --n-w 2048 --single-thread --tag _myqe

`--workers <= 1` runs the generation in-process; `--workers > 1` re-invokes THIS module once per shard
as a subprocess (so each worker gets a fresh JAX JIT cache and -- with --single-thread -- one core),
auto-incrementing the batch index from existing `_batch*` files so reruns ADD batches.  The RES VEGAS
grid is warmed up ONCE up front (a tiny shard) before the workers fan out so they don't race to build it;
the workers then load it from cache.  Total events = workers x seeds-per-worker x n-per-seed.

The generation logic itself lives in `adonis.workflow.generate` (run_generation); this file is only the
thin argparse + parallel-orchestration shell.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

from adonis.workflow.config import load_gen_config
from adonis.workflow.generate import _CHAN_OUT, run_generation


def _bankpath(gc, chanout, tag):
    return os.path.join(gc.out_dir, f"{gc.flux}_{gc.material}_{chanout}{tag}.npz")


def _apply_overrides(gc, a):
    """Apply the per-shard overrides onto a loaded GenConfig (shared by the single-shard path)."""
    if a.no_fsi:                 gc.fsi = False
    if a.n_per_seed is not None: gc.n_per_seed = a.n_per_seed
    if a.n_seeds is not None:    gc.n_seeds = a.n_seeds
    if a.seed0 is not None:      gc.seed0 = a.seed0
    if a.tag is not None:        gc.tag = a.tag
    if a.P is not None:          gc.cascade.P = a.P
    if a.vegas_cache is not None and getattr(gc, "vegas", None) is not None:
        gc.vegas.cache = a.vegas_cache
    return gc


def _run_single(a):
    """One process: run generation in-process (the former adonis_generate.py)."""
    gc = _apply_overrides(load_gen_config(a.config), a)
    out = run_generation(gc, n_w=a.n_w)
    print("DONE:", out, flush=True)


def _run_batched(a):
    """K persistent workers, each re-invoking this module for one shard (the former batched driver)."""
    gc = load_gen_config(a.config)
    K, M, N = a.workers, a.seeds_per_worker, a.n_per_seed
    if a.tag is not None:
        gc.tag = a.tag
    if a.no_fsi:
        gc.tag = gc.tag + "_nofsi"
    p_arg = [] if a.P is None else ["--P", str(a.P)]
    extra = ["--no-fsi"] if a.no_fsi else []
    vegas_on = getattr(gc, "vegas", None) is not None and gc.vegas.enabled

    env = dict(os.environ)
    if a.single_thread:
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
        env["OMP_NUM_THREADS"] = "1"
    me = [sys.executable, "-u", "-m", "adonis.workflow.cli"]     # re-invoke THIS module for each shard

    print(f"[batched] config={a.config} channels={gc.channels} workers={K} seeds/worker={M} "
          f"n/seed={N} -> {K*M*N} events total  P={a.P if a.P is not None else 'config-default'} "
          f"single_thread={a.single_thread} fsi={not a.no_fsi}", flush=True)

    for channel in gc.channels:
        chanout = _CHAN_OUT[channel]
        # 1. VEGAS grid (RES only): build/cache ONCE before the parallel workers (else they race).
        if vegas_on and channel == "res":
            print(f"[batched] {channel}: ensuring VEGAS grid (cache=auto warm-up) ...", flush=True)
            subprocess.run(me + [a.config, "--n-seeds", "1", "--n-per-seed", "2000",
                                 "--seed0", "999000", "--tag", "_gridwarm", "--n-w", "0",
                                 "--vegas-cache", "auto"] + p_arg, check=True, env=env)
            gw = _bankpath(gc, chanout, "_gridwarm")
            if os.path.exists(gw):
                os.remove(gw)
        # 2. auto-increment batch index from existing batch files
        existing = glob.glob(os.path.join(gc.out_dir, f"{gc.flux}_{gc.material}_{chanout}{gc.tag}_batch*.npz"))
        idxs = [int(m.group(1)) for f in existing for m in [re.search(r"_batch(\d+)\.npz$", f)] if m]
        start = (max(idxs) + 1) if idxs else 0
        print(f"[batched] {channel}: {len(existing)} existing batch(es); starting at batch {start}", flush=True)

        # 3. spawn K persistent workers; each runs M seeds in ONE process (JIT once) -> its own _batchWW.npz
        procs, tags = [], []
        for w in range(K):
            idx = start + w
            seed0 = gc.seed0 + idx * M                              # disjoint contiguous seed block per worker
            tag = f"{gc.tag}_batch{idx:02d}"; tags.append(tag)
            cache = "load" if (vegas_on and channel == "res") else "auto"
            procs.append(subprocess.Popen(
                me + [a.config, "--n-seeds", str(M), "--n-per-seed", str(N),
                      "--seed0", str(seed0), "--tag", tag, "--n-w", str(a.n_w),
                      "--vegas-cache", cache] + p_arg + extra,
                stdout=open(f"/tmp/worker_{chanout}_b{idx:02d}.log", "w"), stderr=subprocess.STDOUT, env=env))
        rcs = [p.wait() for p in procs]
        paths = [_bankpath(gc, chanout, t) for t in tags]
        ok = [r == 0 and os.path.exists(p) for r, p in zip(rcs, paths)]
        if not all(ok):
            print(f"[batched] {channel}: WORKER FAILURE rcs={rcs} exists={[os.path.exists(p) for p in paths]}"
                  f" -- see /tmp/worker_{chanout}_b*.log", flush=True)
            sys.exit(1)
        print(f"[batched] {channel}: {K} workers done -> {K} batch files ({M} seeds x {N} ev each)", flush=True)
    print("BATCHED DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ADoNIS forward generation (single process or K parallel workers).")
    ap.add_argument("config")
    # parallelism (workers<=1 -> in-process; workers>1 -> spawn this module per shard)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds-per-worker", type=int, default=1, help="seeds each worker runs in ONE process")
    ap.add_argument("--single-thread", action="store_true", help="1 XLA/Eigen thread per worker (K workers ~ K cores)")
    # generation knobs (shared single + per-shard)
    ap.add_argument("--n-per-seed", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)           # single-process / per-shard seed count
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--n-w", type=int, default=None, help="cascade refill working set (0 = full-batch lock-step)")
    ap.add_argument("--vegas-cache", type=str, default=None, choices=[None, "auto", "load", "rebuild"])
    ap.add_argument("--no-fsi", action="store_true", help="PRE-FSI bank (primary products, no cascade)")
    a = ap.parse_args(argv)

    if a.workers and a.workers > 1:
        if a.n_per_seed is None:
            ap.error("--workers > 1 requires --n-per-seed")
        _run_batched(a)
    else:
        _run_single(a)


if __name__ == "__main__":
    main()
