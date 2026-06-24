"""ADoNIS BATCHED + SHARDED generation driver (the new standard pipeline).

Three knobs: number of BATCHES, events per BATCH, workers per batch.  Batches run SEQUENTIALLY; each
batch is sharded across K worker PROCESSES (events_per_batch // K events each, full-batch cascade at the
new P=2 standard), then the K shards are merged into one per-batch bank `..._batchNN.npz` (w averaged
over the K shards, i.e. an events_per_batch-event estimate of sigma).  The batch index AUTO-INCREMENTS
from existing files, and every (batch, worker) gets a distinct seed -> sequential runs add NEW batches
(no duplicate output).  Reuses scripts/adonis_generate.py (one shard = one --n-seeds 1 invocation) so all
the production setup (target, spectral fns, VEGAS grid, rich record) is shared, not re-implemented.

Usage: python -u scripts/adonis_generate_batched.py configs/gen_c_res_cv5.yaml [--n-batches 10]
       [--events-per-batch 100000] [--workers 4] [--P 2]
"""
import os, sys, glob, argparse, subprocess, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.workflow.config import load_gen_config
from adonis.workflow.generate import _CHAN_OUT

HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(os.path.dirname(HERE), ".venv/bin/python")
GEN = os.path.join(HERE, "adonis_generate.py")


def bankpath(out_dir, chanout, tag):
    return os.path.join(out_dir, f"t2k_{chanout}_engine_rich{tag}.npz")


def merge_shards(shard_paths, out_path, K):
    """Concatenate K shard banks (each ~events_per_batch/K events) into one batch bank; w /= K so the
    merged sum(w) estimates sigma over the full batch (same normalization as the old seed-averaging)."""
    parts = [dict(np.load(p, allow_pickle=True)) for p in shard_paths]
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    bank["w"] = bank["w"] / K
    np.savez(out_path, **bank)
    return len(bank["w"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--events-per-batch", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--no-fsi", action="store_true")      # PRE-FSI banks (no cascade); tag -> _nofsi
    a = ap.parse_args()
    gc = load_gen_config(a.config); K = a.workers; nper = a.events_per_batch // K
    if a.no_fsi:
        gc.tag = gc.tag + "_nofsi"                        # separate bank family from the FSI batches
    vegas_on = getattr(gc, "vegas", None) is not None and gc.vegas.enabled
    extra = ["--no-fsi"] if a.no_fsi else []
    print(f"[batched] config={a.config} channels={gc.channels} n_batches={a.n_batches} "
          f"events/batch={a.events_per_batch} workers={K} ({nper}/worker) P={a.P} fsi={not a.no_fsi}", flush=True)

    for channel in gc.channels:
        chanout = _CHAN_OUT[channel]
        # 1. VEGAS grid (RES only): build/cache ONCE before the parallel workers (else they race), via a
        #    tiny warm-up invocation (cache=auto builds-or-loads); workers then use cache=load.
        if vegas_on and channel == "res":
            print(f"[batched] {channel}: ensuring VEGAS grid (cache=auto warm-up) ...", flush=True)
            subprocess.run([PY, "-u", GEN, a.config, "--n-seeds", "1", "--n-per-seed", "2000",
                            "--seed0", "999000", "--tag", "_gridwarm", "--P", str(a.P), "--n-w", "0",
                            "--vegas-cache", "auto"], check=True)
            gw = bankpath(gc.out_dir, chanout, "_gridwarm")
            if os.path.exists(gw):
                os.remove(gw)
        # 2. auto-increment batch index from existing batch files
        existing = glob.glob(os.path.join(gc.out_dir, f"t2k_{chanout}_engine_rich{gc.tag}_batch*.npz"))
        idxs = [int(m.group(1)) for f in existing for m in [re.search(r"_batch(\d+)\.npz$", f)] if m]
        start = (max(idxs) + 1) if idxs else 0
        print(f"[batched] {channel}: {len(existing)} existing batch(es); starting at batch {start}", flush=True)

        for b in range(start, start + a.n_batches):
            procs, shard_tags = [], []
            for w in range(K):
                seed = gc.seed0 + b * K + w                       # unique per (batch, worker)
                tag = f"{gc.tag}_shard_b{b:02d}_w{w}"; shard_tags.append(tag)
                cache = "load" if (vegas_on and channel == "res") else "auto"
                procs.append(subprocess.Popen(
                    [PY, "-u", GEN, a.config, "--n-seeds", "1", "--n-per-seed", str(nper),
                     "--seed0", str(seed), "--tag", tag, "--P", str(a.P), "--n-w", "0",
                     "--vegas-cache", cache] + extra,
                    stdout=open(f"/tmp/shard_{chanout}_b{b:02d}_w{w}.log", "w"), stderr=subprocess.STDOUT))
            rcs = [p.wait() for p in procs]
            shard_paths = [bankpath(gc.out_dir, chanout, t) for t in shard_tags]
            ok = [r == 0 and os.path.exists(p) for r, p in zip(rcs, shard_paths)]
            if not all(ok):
                print(f"[batched] {channel} batch {b}: WORKER FAILURE rcs={rcs} exists={[os.path.exists(p) for p in shard_paths]}"
                      f" -- see /tmp/shard_{chanout}_b{b:02d}_w*.log", flush=True)
                sys.exit(1)
            out = bankpath(gc.out_dir, chanout, f"{gc.tag}_batch{b:02d}")
            nev = merge_shards(shard_paths, out, K)
            for p in shard_paths:
                os.remove(p)
            print(f"[batched] {channel} batch {b}: merged {K} shards -> {out}  ({nev} events)", flush=True)
    print("BATCHED DONE", flush=True)


if __name__ == "__main__":
    main()
