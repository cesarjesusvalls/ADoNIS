"""Build + freeze a RES VegasGrid for a material and save it as a reproducible sidecar.
Decouples warm-up from generation: `run_channel` will reuse a saved grid if present, or build inline.
Usage: python -u scripts/build_vegas_grid.py [C|Ar] [out.npz] [nbins] [iters] [warmup_n]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec as R
from adonis.xsec.spectral import SpectralFunction

mat = sys.argv[1] if len(sys.argv) > 1 else "C"
out = sys.argv[2] if len(sys.argv) > 2 else f"data/oracle/res_vegasgrid_{mat}.npz"
nbins = int(sys.argv[3]) if len(sys.argv) > 3 else 50
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 6
warmn = int(sys.argv[5]) if len(sys.argv) > 5 else 100000

tg = [t for t, _ in resolve_targets(mat) if t.runs_cascade][0]
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
print(f"warm-up VegasGrid for {tg.symbol}{tg.A}: {iters} iters x {warmn} ev, nbins={nbins}", flush=True)
grid = R.warmup_vegas(n=warmn, iters=iters, nbins=nbins, sf_n=sf_n, sf_p=sf_p,
                      n_neutron=tg.A - tg.Z, n_proton=tg.Z)
grid.save(out)
print(f"wrote {out}", flush=True)
