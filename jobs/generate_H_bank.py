"""Shim kept so the S3DF submitters keep working; the tool is adonis.workflow.generate_h_bank,
reachable through adonis.workflow.cli with a free-nucleon config.

    python jobs/generate_H_bank.py <outdir> <n_per_chunk> <n_chunks> [seed0]

The flux is the module default here.  Prefer the config-driven form, where
the flux comes from the bank config and is recorded in the manifest.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.workflow.config import load_gen_config
from adonis.workflow.generate_h_bank import generate

if __name__ == "__main__":
    out, n_per, n_chunks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = load_gen_config(str(Path(__file__).resolve().parents[1] / "configs/banks/nu_T2K_H.yaml"))
    cfg.n_per_seed, cfg.n_seeds = n_per, n_chunks
    cfg.seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    generate(cfg, out)
