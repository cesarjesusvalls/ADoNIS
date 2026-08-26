"""Combine per-shard extractor output into one reference npz.

Each ACHILLES shard is extracted separately, so its weights carry that shard's own hepmc normalisation.
Weights are converted to absolute nb here before concatenation and the combined file records
weight_to_nb = 1/n_shards, which is what averaging N independent estimates of the same cross section
leaves to apply.  A shard that did not store its normalisation has it read back from its hepmc.

Fields that vary per event are concatenated; scalars are dropped, since they describe one shard.

    python -m analysis.oracle_tools.combine "output/achilles/nu_T2K_C/*.fs_rich.npz" \\
        output/achilles/fsrich/nu_T2K_C.npz
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

from adonis.oracle.hepmc import hepmc_norm


def _weight_to_nb(path, d):
    if "weight_to_nb" in d.files:
        return float(d["weight_to_nb"])
    hepmc = path.rsplit(".fsrich.npz", 1)[0].rsplit(".npz", 1)[0] + ".hepmc"
    return float(hepmc_norm(hepmc)["weight_to_nb"])


def combine(pattern, out):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no extractor output matching {pattern}")

    acc, absolute = {}, []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            scale = _weight_to_nb(f, d)
            for k in d.files:
                if k == "w" or np.asarray(d[k]).ndim == 0:
                    continue
                acc.setdefault(k, []).append(d[k])
            absolute.append(np.asarray(d["w"]) * scale)

    o = {k: np.concatenate(v) for k, v in acc.items()}
    o["w"] = np.concatenate(absolute)
    o["weight_to_nb"] = 1.0 / len(files)
    o["n_shards"] = len(files)
    np.savez(out, **o)
    print(f"  {len(files)} shards, {len(o['w']):,} events -> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.oracle_tools.combine",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("pattern", help="glob over the per-shard npz (quote it)")
    ap.add_argument("out", help="combined npz")
    a = ap.parse_args(argv)
    combine(a.pattern, a.out)


if __name__ == "__main__":
    main()
