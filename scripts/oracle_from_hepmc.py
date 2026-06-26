"""Build the final-state oracle npz from one or more ACHILLES hepmc files.

Used by CI after running the published container image to produce the hepmc:
  python scripts/oracle_from_hepmc.py out.npz events1.hepmc [events2.hepmc ...]

Histograms each file (sum w, sum w^2 per bin) on the shared ORACLE_EDGES and
saves the combined result -- identical binning to the local high-statistics
generator (scripts/make_oracle_finalstate.py), so the model/oracle comparison and
the per-module oracle_test consume the same format regardless of how it was made.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys

from analysis.utils.finalstate import new_accumulator, accumulate_hepmc, save_oracle

if len(sys.argv) < 3:
    sys.exit("usage: oracle_from_hepmc.py OUT.npz HEPMC [HEPMC ...]")

out = sys.argv[1]
hepmcs = sys.argv[2:]
acc = new_accumulator()
n_total = 0
for h in hepmcs:
    n = accumulate_hepmc(h, acc)
    n_total += n
    print(f"  {h}: {n:,} signal events  (total {n_total:,})", flush=True)

save_oracle(out, n_total, acc)
print(f"saved -> {out}  ({n_total:,} total signal events)")
