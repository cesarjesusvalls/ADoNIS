import sys, glob, numpy as np
sys.path.insert(0, "/sdf/home/c/cjesus/DIFFGEN/ADoNIS")
from adonis.oracle.hepmc import hepmc_norm
pat, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(pat, recursive=True))
if not files: sys.exit(f"no fsrich npz for {pat}")
acc = {}; absw = []; N = 0
for f in files:
    d = np.load(f, allow_pickle=True)
    if "weight_to_nb" in d.files:                       # extract.py already stored the norm (scalar)
        wtnb = float(d["weight_to_nb"])
    else:                                               # my reext_fs.py did not -> read the hepmc header
        hep = f.rsplit(".fsrich.npz", 1)[0].rsplit(".npz", 1)[0] + ".hepmc"
        wtnb = float(hepmc_norm(hep)["weight_to_nb"])
    for k in d.files:
        if k == "w" or np.asarray(d[k]).ndim == 0:      # skip w (handled) + scalar meta (gen_xs_pb, ...)
            continue
        acc.setdefault(k, []).append(d[k])
    absw.append(np.asarray(d["w"]) * wtnb)
    N += 1
o = {k: np.concatenate(v) for k, v in acc.items()}
o["w"] = np.concatenate(absw); o["weight_to_nb"] = 1.0 / N; o["n_shards"] = N
np.savez(out, **o)
print(f"  {N} shards, {len(o['w']):,} events -> {out}")
