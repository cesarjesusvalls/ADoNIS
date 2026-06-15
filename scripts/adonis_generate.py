"""ADoNIS generation driver: generate engine rich bank(s) from a YAML config.

Usage: python -u scripts/adonis_generate.py configs/gen_cc1pi.yaml

Sets ADONIS_N_RECOIL from the config's cascade.n_recoil BEFORE importing the cascade (which reads it
at import time), then runs the generation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

cfg_path = sys.argv[1]
_d = yaml.safe_load(open(cfg_path)) or {}
os.environ["ADONIS_N_RECOIL"] = str((_d.get("cascade") or {}).get("n_recoil", 4))  # before cascade import

from adonis.workflow.config import load_gen_config       # noqa: E402
from adonis.workflow.generate import run_generation       # noqa: E402

if __name__ == "__main__":
    out = run_generation(load_gen_config(cfg_path))
    print("DONE:", out)
