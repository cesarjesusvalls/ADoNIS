"""ADoNIS generation driver: generate engine rich bank(s) from a YAML config.

Usage: python -u scripts/adonis_generate.py configs/gen_cc1pi.yaml [overrides]
Overrides (used by the batched/sharded driver -- adonis_generate_batched.py -- to run one shard):
  --n-per-seed N   --n-seeds N   --seed0 N   --tag STR   --P N   --n-w N   --vegas-cache auto|load|rebuild
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adonis.workflow.config import load_gen_config       # noqa: E402
from adonis.workflow.generate import run_generation       # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-per-seed", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--n-w", type=int, default=None)      # cascade refill working set (0 = full-batch)
    ap.add_argument("--vegas-cache", type=str, default=None)
    ap.add_argument("--no-fsi", action="store_true")      # PRE-FSI bank (primary products, no cascade)
    a = ap.parse_args()
    gc = load_gen_config(a.config)
    if a.no_fsi:                 gc.fsi = False
    if a.n_per_seed is not None: gc.n_per_seed = a.n_per_seed
    if a.n_seeds is not None:    gc.n_seeds = a.n_seeds
    if a.seed0 is not None:      gc.seed0 = a.seed0
    if a.tag is not None:        gc.tag = a.tag
    if a.P is not None:          gc.cascade.P = a.P
    if a.vegas_cache is not None and getattr(gc, "vegas", None) is not None:
        gc.vegas.cache = a.vegas_cache
    out = run_generation(gc, n_w=a.n_w)
    print("DONE:", out)
