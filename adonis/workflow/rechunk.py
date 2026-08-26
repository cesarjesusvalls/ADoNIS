"""Repack a bank into uniform, memory-balanced chunks.

Banks can end up chunked unevenly, and streaming cost is dominated by per-chunk overhead, so this tool
rewrites a bank so every chunk loads to ~the same memory footprint (default 0.5 GB), preserving the
full record schema and the cross-section normalization.

Normalization: the bank convention is  cross_section = sum(stored w0) / n_chunks  (each chunk a
1/n_chunks estimate).  Repacking into M chunks stores  w0_final * M  and sets manifest n_chunks = M, so
load_bank(dst) reproduces load_bank(src) exactly.

    python -m adonis.workflow.rechunk output/beam_e_C_hv/merged output/beam_e_C_hv_rechunk/merged --target-gb 0.5

Loads the whole source bank into memory once; writes stream out.
"""
import argparse
import json
import os

import numpy as np

from adonis.reweight.bank_plot import load_bank, filter_events


def rechunk(src, dst, target_bytes=0.5e9, log=print):
    B = load_bank(src)
    n = len(B["w0"])
    total = sum(np.asarray(v).nbytes for v in B.values() if isinstance(v, np.ndarray))
    M = max(1, int(round(total / target_bytes)))
    edges = np.linspace(0, n, M + 1).astype(int)
    os.makedirs(dst, exist_ok=True)
    log(f"repack {src}: {n:,} events, {total/1e9:.2f} GB -> {M} chunks (~{total/max(M,1)/1e9:.2f} GB each)")
    for i in range(M):
        m = np.zeros(n, bool); m[edges[i]:edges[i + 1]] = True
        Bc = filter_events(B, m)
        out = {k: v for k, v in Bc.items() if k not in ("_eidx", "n_chunks", "labels")}
        out["w0"] = np.asarray(Bc["w0"]) * M
        np.savez(f"{dst}/chunk_{i:04d}.npz", **out)
    man = {}
    try:
        man = json.load(open(f"{src}/manifest.json"))
    except FileNotFoundError:
        pass
    man["n_chunks"] = M
    json.dump(man, open(f"{dst}/manifest.json", "w"))
    log(f"[out] {dst}  ({M} chunks, manifest n_chunks={M})")
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="source bank dir (…/merged)")
    ap.add_argument("dst", help="destination bank dir (…/merged)")
    ap.add_argument("--target-gb", type=float, default=0.5, help="target chunk size in GB (default 0.5)")
    a = ap.parse_args()
    rechunk(a.src, a.dst, target_bytes=a.target_gb * 1e9)
