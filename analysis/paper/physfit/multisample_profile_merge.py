"""Merge per-dial profile shards (<label>_profile_shNN.npz) into <label>_profile.npz.

Each shard filled only its own dial's row and left the rest NaN, so the merge is a NaN-aware overlay.
Reports any dial that no shard covered rather than silently writing a NaN row.
"""
import sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

def main(label="sec4_all16"):
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_profile_sh*.npz")))
    if not fs:
        raise SystemExit(f"no shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    base = Z[0]
    prof = np.full_like(np.asarray(base["prof_dobj"]), np.nan)
    profd = np.full_like(prof, np.nan)
    grids = np.full_like(prof, np.nan)
    ldet = np.full_like(prof, np.nan)
    for z in Z:
        keys = [(prof, "prof_dobj"), (profd, "prof_dchi2"), (grids, "grids_sigma")]
        if "logdet_Vnuis" in z.files:
            keys.append((ldet, "logdet_Vnuis"))
        for arr, key in keys:
            v = np.asarray(z[key]); m = np.isfinite(v).any(axis=1)
            arr[m] = v[m]
    pn = [str(x) for x in base["pnames"]]; sub = [int(k) for k in base["subset"]]
    missing = [pn[sub[c]] for c in range(len(sub)) if not np.isfinite(prof[c]).any()]
    out = style.ALTGEN / f"{label}_profile.npz"
    np.savez(out, subset=sub, pnames=base["pnames"], grid_sigma=base["grid_sigma"],
             grids_sigma=grids, prof_dobj=prof, prof_dchi2=profd, logdet_Vnuis=ldet,
             bfp=base["bfp"],
             sigma_post=base["sigma_post"], truth=base["truth"], chi2_min=base["chi2_min"])
    print(f"merged {len(fs)} shards -> {out}")
    print(f"dials covered: {len(sub)-len(missing)}/{len(sub)}" + (f"   MISSING: {missing}" if missing else ""))

if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
