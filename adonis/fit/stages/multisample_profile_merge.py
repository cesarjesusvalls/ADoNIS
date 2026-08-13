"""Merge per-dial profile shards (<label>_profile_shNN.npz) into <label>_profile.npz.

Each shard filled only its own dial's row and left the rest NaN, so the merge is a NaN-aware overlay.

VALIDATED before merging (adonis.fit.merge): the shards must come from one run definition, and the dials
they were ASSIGNED must tile the fitted subset.  An uncovered dial used to be reported as a line of
stdout on the way to writing the file anyway; it is now a refusal, because the merged npz is what every
downstream figure reads and a NaN row there is indistinguishable from a dial that legitimately has no
profile.  --allow-partial accepts it knowingly, and marks the product partial.
"""
import sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from adonis.fit import merge as MG

def main(label="sec4_all16", allow_partial=False):
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_profile_sh*.npz")))
    if not fs:
        raise SystemExit(f"no shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    rep = MG.check(fs, Z, what=f"{label} profile")
    # Coverage over the DIAL index, from what each shard says it was assigned.  Shards written before
    # stamping say nothing, and then there is nothing to check but the NaN rows below.
    blocks = [(p["dial_base"], p["n_dial"]) for p in (MG.provenance.read(z) for z in Z)
              if p and "dial_base" in p]
    if blocks:
        rep.rows(blocks, grid=int(np.max([p["n_subset"] for p in
                                          (MG.provenance.read(z) for z in Z) if p and "n_subset" in p])))
    rep.raise_if_bad(allow_partial)
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
    # The DATA-level check, kept alongside the assignment-level one: a dial can be assigned to a shard
    # that ran and still produce nothing.
    missing = [pn[sub[c]] for c in range(len(sub)) if not np.isfinite(prof[c]).any()]
    if missing:
        rep.error(f"no profile curve for {len(missing)} dial(s): {missing}")
        rep.raise_if_bad(allow_partial)
    out = style.ALTGEN / f"{label}_profile.npz"
    np.savez(out, **rep.stamp(), subset=sub, pnames=base["pnames"], grid_sigma=base["grid_sigma"],
             grids_sigma=grids, prof_dobj=prof, prof_dchi2=profd, logdet_Vnuis=ldet,
             bfp=base["bfp"],
             sigma_post=base["sigma_post"], truth=base["truth"], chi2_min=base["chi2_min"])
    print(f"merged {len(fs)} shards -> {out}")
    print(f"dials covered: {len(sub)-len(missing)}/{len(sub)}" + (f"   MISSING: {missing}" if missing else ""))

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(*(args[:1] or []), allow_partial="--allow-partial" in sys.argv)
