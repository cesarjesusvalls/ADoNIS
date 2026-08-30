"""Merge sharded banks into one bank directory.

Generation shards independently: each shard writes <parts>/part_<id>/chunk_*.npz + manifest.json under
its own seeds.  This collects every shard's chunks into <out>/chunk_<n>.npz (hardlinked, or symlinked
across filesystems) and writes a merged manifest.

load_bank divides the stored w0 by manifest["n_chunks"], so n_chunks must equal the number of chunk
files present; a shard whose chunk is corrupt or half-written is skipped and not counted.  All shards
must have been generated with the same events-per-chunk for that normalisation to be uniform.

    python -m adonis.workflow.merge_bank --parts output/nu_T2K_C --out output/nu_T2K_C/merged
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def link(src, dst):
    """Hardlink src to dst, falling back to a symlink across filesystems."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def _shard_dirs(parts, pattern):
    return sorted(glob.glob(str(Path(parts) / pattern)),
                  key=lambda p: int("".join(filter(str.isdigit, Path(p).name)) or -1))


def _events_in(chunk):
    """Event count of one chunk, or None if it cannot be read."""
    try:
        with np.load(chunk) as d:
            key = next((k for k in ("w0", "c", "weight") if k in d.files), d.files[0])
            return len(d[key])
    except Exception as e:                       # noqa: BLE001 - a partial shard must not abort the merge
        print(f"  SKIP unreadable {chunk}: {e}", flush=True)
        return None


def merge(parts, out, pattern="part_*"):
    """Link every shard chunk into `out` and write the merged manifest.  Returns the chunk count."""
    part_dirs = _shard_dirs(parts, pattern)
    if not part_dirs:
        raise SystemExit(f"no shard directories matching {parts}/{pattern}")

    manifest = next((json.loads((Path(p) / "manifest.json").read_text())
                     for p in part_dirs if (Path(p) / "manifest.json").exists()), None)
    if manifest is None:
        raise SystemExit(f"no shard under {parts} has a manifest.json to take the constant fields from")

    os.makedirs(out, exist_ok=True)
    idx = n_total = 0
    for pd in part_dirs:
        for c in sorted(glob.glob(str(Path(pd) / "chunk_*.npz"))):
            nev = _events_in(c)
            if nev is None:
                continue
            link(os.path.abspath(c), str(Path(out) / f"chunk_{idx:03d}.npz"))
            idx += 1
            n_total += nev
    if idx == 0:
        raise SystemExit(f"no readable chunks under {parts}/{pattern}")

    manifest["n_chunks"] = idx
    manifest["n_total"] = n_total
    manifest["merged_from"] = len(part_dirs)
    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"merged {idx} chunks from {len(part_dirs)} shards -> {out}  (n_total={n_total:,})", flush=True)
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adonis.workflow.merge_bank", description=__doc__.split("\n")[0])
    ap.add_argument("--parts", required=True, help="directory holding the per-shard subdirectories")
    ap.add_argument("--out", required=True, help="merged bank directory")
    ap.add_argument("--glob", default="part_*", dest="pattern", help="shard subdirectory pattern")
    a = ap.parse_args(argv)
    merge(a.parts, a.out, a.pattern)


if __name__ == "__main__":
    main()
