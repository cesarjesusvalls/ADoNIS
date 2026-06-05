"""Write one ACHILLES oracle-batch run config (used by the CI oracle workflow).

  python scripts/write_oracle_config.py BASE.yml OUT.yml NEVENTS SEED OUT_HEPMC
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys

from adonis.data.oracle.finalstate import write_run_config

if len(sys.argv) != 6:
    sys.exit("usage: write_oracle_config.py BASE.yml OUT.yml NEVENTS SEED OUT_HEPMC")
base, out, nevents, seed, out_hepmc = sys.argv[1:]
write_run_config(base, out, int(nevents), int(seed), out_hepmc)
print(f"wrote {out}  (NEvents={nevents}, Seed={seed}, Output={out_hepmc})")
