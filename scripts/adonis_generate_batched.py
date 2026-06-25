"""ADoNIS PERSISTENT-WORKER generation driver (the standard pipeline).

K worker PROCESSES are spawned ONCE and run in parallel; each worker does M seeds of N events in a single
process, so JAX JIT-compiles the cascade kernel ONCE per worker and reuses it across that worker's seeds
(not once per seed).  Each worker writes its OWN final bank `..._batchWW.npz` directly, checkpointed
after every seed by run_channel (w normalized by seeds-banked-so-far) -- so every batch file is a valid,
correctly-normalized partial sigma estimate AT ALL TIMES.  The analysis scripts glob `_batch*` and
average the K files; averaging K unbiased sigma estimates is unbiased even while workers are mid-run, so
you can plot live as seeds complete (no separate merge step).

Total events = workers x seeds-per-worker x n-per-seed, in K bank files of (M*N) events each.
Worker w gets a disjoint contiguous seed block, and the batch index auto-increments from existing files,
so sequential runs ADD new batches.

--single-thread caps each worker to one XLA/Eigen thread (xla_cpu_multi_thread_eigen=false, OMP=1) so K
single-threaded workers fit K cores without oversubscribing (vs the default where each worker spawns a
full thread pool -> K x ncpu threads on ncpu cores).

Usage: python -u scripts/adonis_generate_batched.py configs/gen_c_qe_cv5.yaml
       [--workers 6] [--seeds-per-worker 2] [--n-per-seed 10000] [--P 10] [--single-thread] [--no-fsi]
"""
import os, sys, glob, argparse, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adonis.workflow.config import load_gen_config
from adonis.workflow.generate import _CHAN_OUT

HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(os.path.dirname(HERE), ".venv/bin/python")
GEN = os.path.join(HERE, "adonis_generate.py")


def bankpath(out_dir, chanout, tag):
    return os.path.join(out_dir, f"t2k_{chanout}_engine_rich{tag}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=6)            # persistent worker processes
    ap.add_argument("--seeds-per-worker", type=int, default=1)   # seeds each worker runs in ONE process (JIT once)
    ap.add_argument("--n-per-seed", type=int, default=10000)     # events per seed
    ap.add_argument("--P", type=int, default=None)              # None -> inherit config / CascadeHyperparams default
    ap.add_argument("--single-thread", action="store_true")      # 1 XLA/Eigen thread per worker (K workers ~ K cores)
    ap.add_argument("--no-fsi", action="store_true")             # PRE-FSI banks (no cascade); tag -> _nofsi
    a = ap.parse_args()
    gc = load_gen_config(a.config); K, M, N = a.workers, a.seeds_per_worker, a.n_per_seed
    p_arg = [] if a.P is None else ["--P", str(a.P)]            # pass-through only if explicitly set
    if a.no_fsi:
        gc.tag = gc.tag + "_nofsi"
    vegas_on = getattr(gc, "vegas", None) is not None and gc.vegas.enabled
    extra = ["--no-fsi"] if a.no_fsi else []
    env = dict(os.environ)
    if a.single_thread:
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
        env["OMP_NUM_THREADS"] = "1"
    print(f"[batched] config={a.config} channels={gc.channels} workers={K} seeds/worker={M} "
          f"n/seed={N} -> {K*M*N} events total  P={a.P if a.P is not None else 'config-default'} "
          f"single_thread={a.single_thread} fsi={not a.no_fsi}",
          flush=True)

    for channel in gc.channels:
        chanout = _CHAN_OUT[channel]
        # 1. VEGAS grid (RES only): build/cache ONCE before the parallel workers (else they race).
        if vegas_on and channel == "res":
            print(f"[batched] {channel}: ensuring VEGAS grid (cache=auto warm-up) ...", flush=True)
            subprocess.run([PY, "-u", GEN, a.config, "--n-seeds", "1", "--n-per-seed", "2000",
                            "--seed0", "999000", "--tag", "_gridwarm", "--n-w", "0",
                            "--vegas-cache", "auto"] + p_arg, check=True, env=env)
            gw = bankpath(gc.out_dir, chanout, "_gridwarm")
            if os.path.exists(gw):
                os.remove(gw)
        # 2. auto-increment batch index from existing batch files
        existing = glob.glob(os.path.join(gc.out_dir, f"t2k_{chanout}_engine_rich{gc.tag}_batch*.npz"))
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
                [PY, "-u", GEN, a.config, "--n-seeds", str(M), "--n-per-seed", str(N),
                 "--seed0", str(seed0), "--tag", tag, "--n-w", "0",
                 "--vegas-cache", cache] + p_arg + extra,
                stdout=open(f"/tmp/worker_{chanout}_b{idx:02d}.log", "w"), stderr=subprocess.STDOUT, env=env))
        rcs = [p.wait() for p in procs]
        paths = [bankpath(gc.out_dir, chanout, t) for t in tags]
        ok = [r == 0 and os.path.exists(p) for r, p in zip(rcs, paths)]
        if not all(ok):
            print(f"[batched] {channel}: WORKER FAILURE rcs={rcs} exists={[os.path.exists(p) for p in paths]}"
                  f" -- see /tmp/worker_{chanout}_b*.log", flush=True)
            sys.exit(1)
        print(f"[batched] {channel}: {K} workers done -> {K} batch files ({M} seeds x {N} ev each)", flush=True)
    print("BATCHED DONE", flush=True)


if __name__ == "__main__":
    main()
