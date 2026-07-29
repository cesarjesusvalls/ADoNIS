#!/usr/bin/env python3
"""Merge sharded chunk-banks (event_bank / beam_bank) into one bank dir.

Each SLURM array shard writes <parts>/part_<id>/chunk_000.npz + manifest.json (independent seeds via
SEED0/--seed). This collects every shard's chunk_*.npz into <out>/chunk_<globalidx:03d>.npz (hardlink
when possible, else symlink) and writes a merged manifest with n_chunks = total chunk count and
n_total = sum of the shards' n_total.

`load_bank` (analysis/t2k/differentiability/bank_plot.py) globs chunk_*.npz and divides w0 by
manifest["n_chunks"], so the merged n_chunks MUST equal the number of chunk files -- which it does.
All shards must share the same per-chunk size (CHUNK) for the w0/CHUNK normalization to be uniform.

    python jobs/merge_bank.py --parts $ADONIS_OUT/event_bank_10M --out $ADONIS_OUT/event_bank_10M/merged
"""
import argparse
import glob
import json
import os
from pathlib import Path


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)          # hardlink: no extra space, survives part-dir removal (same fs)
    except OSError:
        os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="dir containing part_*/ shard subdirs")
    ap.add_argument("--out", required=True, help="merged bank dir")
    ap.add_argument("--glob", default="part_*", help="shard subdir glob (default part_*)")
    a = ap.parse_args()

    part_dirs = sorted(glob.glob(str(Path(a.parts) / a.glob)),
                       key=lambda p: int("".join(filter(str.isdigit, Path(p).name)) or -1))
    if not part_dirs:
        raise SystemExit(f"no shard dirs matching {a.parts}/{a.glob}")
    os.makedirs(a.out, exist_ok=True)

    # A shard preempted mid-run leaves VALID chunks but no manifest.json (beam_bank writes the manifest
    # only at the very end). Each chunk_*.npz is a complete independent sub-bank, so include EVERY
    # loadable chunk from EVERY part; the manifest is only needed for the constant fields (chunk size,
    # caps, pmin/pmax/pir2_mb/species) -- take those from any one part that finished.
    import numpy as np
    base_manifest = None
    per_chunk_events = None
    for pd in part_dirs:
        mf = Path(pd) / "manifest.json"
        if mf.exists():
            base_manifest = dict(json.load(open(mf)))
            break
    if base_manifest is None:
        raise SystemExit("no part has a manifest.json (need one for the constant fields)")

    idx = 0
    n_total = 0
    for pd in part_dirs:
        for c in sorted(glob.glob(str(Path(pd) / "chunk_*.npz"))):
            try:
                with np.load(c) as d:                       # load-test: skip a chunk truncated by a kill
                    kk = next((k for k in ("w0", "c", "weight") if k in d.files), d.files[0])
                    nev = len(d[kk])                         # per-event key: neutrino w0 | (e,e') c | beam
            except Exception as e:
                print(f"  SKIP corrupt/partial {c}: {e}")
                continue
            link(os.path.abspath(c), str(Path(a.out) / f"chunk_{idx:03d}.npz"))
            idx += 1
            n_total += nev
    if idx == 0:
        raise SystemExit("no chunks merged")

    base_manifest["n_chunks"] = idx
    base_manifest["n_total"] = n_total
    base_manifest["merged_from"] = len(part_dirs)
    json.dump(base_manifest, open(Path(a.out) / "manifest.json", "w"), indent=2)
    print(f"merged {idx} chunks from {len(part_dirs)} shards -> {a.out}  (n_total={n_total:,})")


if __name__ == "__main__":
    main()
