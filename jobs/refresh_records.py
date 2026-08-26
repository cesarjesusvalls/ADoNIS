"""Refresh a CC bank's QE+RES hard-vertex records to the exact reduced-quadratic form (docs/joint_amps2_plan.md).

Per chunk, rebuild from STORED kinematics (no event regeneration):
  QE  (channel==0): M from k_nu, k_lep, p_struck, p_out = k_nu + p_struck - k_lep (2-body conservation).
  RES (channel==1): M from k_nu, k_lep, p_struck, res_p_N, res_p_pi, res_ipid, res_ppid (vertex momenta,
                    verified to reproduce the stored records to float32).
The legacy per-knob QE+RES records are DROPPED; hv_qe_mij/hv_qe_Q2 (N,4,4) and hv_res_mij/hv_res_Q2 (N,6,6)
are written (the OTHER channel's rows are 0 -> reduced reweight == 1).  Everything else copied verbatim.

    python -m jobs.refresh_records <src merged dir> <dst dir>        # CC banks only (nu_*)
"""
import os
import sys
import glob
import shutil

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.reweight.reduced_amps2 import build_qe_reduced, build_res_reduced

_DROP = (tuple(f"hv_qe_{r}_" for r in ("ma", "vec", "gmp", "gmn", "gep", "gen"))
         + tuple(f"hv_res_{r}_" for r in ("ma", "pp", "delta"))
         + ("hv_qe_mij", "hv_qe_Q2", "hv_res_mij", "hv_res_Q2"))


def refresh_chunk(inpath, outpath):
    B = dict(np.load(inpath, allow_pickle=True))
    ch = np.asarray(B["channel"]); n = len(ch); qe = ch == 0; res = ch == 1
    lep = "k_mu" if "k_mu" in B else "k_lep"
    kn, km, ps = np.asarray(B["k_nu"]), np.asarray(B[lep]), np.asarray(B["p_struck"])
    Mq = np.zeros((n, 4, 4), np.float32); Q2q = np.zeros(n, np.float32)
    if qe.any():
        p_out = (kn + ps - km)[qe]
        rq = build_qe_reduced(kn[qe], km[qe], ps[qe], p_out, probe="CC")
        Mq[qe] = np.asarray(rq["M"], np.float32); Q2q[qe] = np.asarray(rq["Q2"], np.float32)
    Mr = np.zeros((n, 6, 6), np.float64); Q2r = np.zeros(n, np.float32)
    if res.any():
        rr = build_res_reduced(kn[res], km[res], ps[res], np.asarray(B["res_p_N"])[res],
                               np.asarray(B["res_p_pi"])[res], np.asarray(B["res_ipid"])[res],
                               np.asarray(B["res_ppid"])[res])
        Mr[res] = np.asarray(rr["M"], np.float64); Q2r[res] = np.asarray(rr["Q2"], np.float32)
    out = {k: v for k, v in B.items() if not k.startswith(_DROP)}
    out.update(hv_qe_mij=Mq, hv_qe_Q2=Q2q, hv_res_mij=Mr, hv_res_Q2=Q2r)
    tmp = outpath + ".tmp.npz"
    np.savez(tmp, **out); os.replace(tmp, outpath)
    return int(qe.sum()), int(res.sum()), n


_BANKS = [("nu_T2K_C", "merged_perknob"), ("nu_MINERvA_C", "merged"), ("nu_uBooNE_Ar", "merged")]


def _worklist():
    wl = []
    for bank, src in _BANKS:
        ddir = f"output/paper_banks_p4/{bank}/merged_mij"
        for c in sorted(glob.glob(f"output/paper_banks_p4/{bank}/{src}/chunk_*.npz")):
            wl.append((c, os.path.join(ddir, os.path.basename(c))))
    return wl


def _ensure_dst():
    for bank, src in _BANKS:
        ddir = f"output/paper_banks_p4/{bank}/merged_mij"; os.makedirs(ddir, exist_ok=True)
        man, dman = f"output/paper_banks_p4/{bank}/{src}/manifest.json", os.path.join(ddir, "manifest.json")
        if os.path.exists(man) and not os.path.exists(dman):
            shutil.copy(man, dman)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--shard":
        I, K = int(sys.argv[2]), int(sys.argv[3])
        _ensure_dst()
        mine = _worklist()[I::K]
        done = skip = 0
        for c, op in mine:
            if os.path.exists(op):
                skip += 1; continue
            refresh_chunk(c, op); done += 1
            print(f"[shard {I}%{K}] wrote {os.path.basename(op)}", flush=True)
        print(f"shard {I}/{K}: refreshed {done}, skipped {skip}, of {len(mine)} assigned", flush=True)
        return
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(src, "chunk_*.npz")))
    if not chunks:
        raise SystemExit(f"no chunk_*.npz in {src}")
    nq = nr = nt = skip = 0
    for i, c in enumerate(chunks):
        op = os.path.join(dst, os.path.basename(c))
        if os.path.exists(op):
            skip += 1
            continue
        q, r, t = refresh_chunk(c, op)
        nq += q; nr += r; nt += t
        print(f"[{i + 1}/{len(chunks)}] {os.path.basename(c)}  QE={q} RES={r} /{t}", flush=True)
    print(f"(resumed: {skip} chunks already present)", flush=True)
    man = os.path.join(src, "manifest.json")
    if os.path.exists(man):
        shutil.copy(man, os.path.join(dst, "manifest.json"))
    print(f"done: {len(chunks)} chunks, QE={nq} RES={nr} /{nt} refreshed -> {dst}", flush=True)


if __name__ == "__main__":
    main()
