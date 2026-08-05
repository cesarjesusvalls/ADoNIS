"""Refresh a CC bank's QE hard-vertex records to the exact reduced-quadratic form (docs/joint_amps2_plan.md).

Per chunk: rebuild the QE reduced record M from the STORED kinematics -- k_nu, k_mu(=k_lep), p_struck, and the
vertex outgoing nucleon p_out = k_nu + p_struck - k_lep (2-body conservation; verified to reproduce the stored
per-knob records to float32).  RES events get M=0 (-> qe_reduced_reweight == 1, channel-correct padding).  The 6
legacy QE per-knob records (hv_qe_{ma,vec,gmp,gmn,gep,gen}_*) are DROPPED and replaced by hv_qe_mij + hv_qe_Q2;
everything else (RES records, FSI f_*, w0, kinematics, final state) is copied verbatim.  No event regeneration.

    python -m jobs.refresh_qe_mij <src merged dir> <dst dir>       # CC banks only (nu_*)

The caller does the copy-then-swap (mv src src_perknob; mv dst src) after validation.
"""
import os
import sys
import glob
import json
import shutil

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.reweight.reduced_amps2 import build_qe_reduced

_DROP_PREFIX = tuple(f"hv_qe_{r}_" for r in ("ma", "vec", "gmp", "gmn", "gep", "gen"))


def refresh_chunk(inpath, outpath):
    B = dict(np.load(inpath, allow_pickle=True))
    ch = np.asarray(B["channel"]); n = len(ch); qe = ch == 0
    # the merged bank mixes sub-runs: the outgoing lepton is stored as k_mu in some chunks, k_lep in others.
    lep = "k_mu" if "k_mu" in B else "k_lep"
    kn, km, ps = np.asarray(B["k_nu"]), np.asarray(B[lep]), np.asarray(B["p_struck"])
    p_out = kn + ps - km                                    # vertex outgoing nucleon (2-body conservation)
    M = np.zeros((n, 4, 4), np.float32); Q2 = np.zeros(n, np.float32)
    if qe.any():
        rec = build_qe_reduced(kn[qe], km[qe], ps[qe], p_out[qe], probe="CC")
        M[qe] = np.asarray(rec["M"], np.float32)
        Q2[qe] = np.asarray(rec["Q2"], np.float32)
    out = {k: v for k, v in B.items() if not k.startswith(_DROP_PREFIX)}
    out["hv_qe_mij"] = M; out["hv_qe_Q2"] = Q2
    np.savez(outpath, **out)
    return int(qe.sum()), n


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(src, "chunk_*.npz")))
    if not chunks:
        raise SystemExit(f"no chunk_*.npz in {src}")
    nq = nt = 0
    for i, c in enumerate(chunks):
        q, t = refresh_chunk(c, os.path.join(dst, os.path.basename(c)))
        nq += q; nt += t
        print(f"[{i + 1}/{len(chunks)}] {os.path.basename(c)}  QE={q}/{t}", flush=True)
    man = os.path.join(src, "manifest.json")
    if os.path.exists(man):
        shutil.copy(man, os.path.join(dst, "manifest.json"))
    print(f"done: {len(chunks)} chunks, {nq}/{nt} QE events refreshed -> {dst}", flush=True)


if __name__ == "__main__":
    main()
