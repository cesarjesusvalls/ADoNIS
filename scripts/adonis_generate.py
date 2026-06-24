"""ADoNIS generation driver: generate engine rich bank(s) from a YAML config.

Usage: python -u scripts/adonis_generate.py configs/gen_cc1pi.yaml
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adonis.workflow.config import load_gen_config       # noqa: E402
from adonis.workflow.generate import run_generation       # noqa: E402

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    out = run_generation(load_gen_config(cfg_path))
    print("DONE:", out)
