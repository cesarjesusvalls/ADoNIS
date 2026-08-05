"""Standalone free-proton H bank generator for the T2K CH CC1pi+ contribution.

nu_mu p -> mu- p pi+ on a FREE proton at rest: RES only (no CC-QE on a proton), NO nuclear FSI (pi+ always
survives), NO spectral function (proton at rest).  Produces a schema-compatible bank so it plugs into
bank_signal (sec1 CH figures) and is reweightable via res_reduced_reweight (sec2/3): the ragged final state
(fs_*), w0, the EXACT reduced-quadratic RES hard-vertex record (hv_res_mij, float64), zero QE record, and
EMPTY FSI records (-> FSI reweight == 1).  No shared generation code touched -> existing banks untouched.

    python -m jobs.generate_H_bank <outdir> <n_per_chunk> <n_chunks> [seed0]
"""
import os
import sys
import json

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.channels.free_proton import generate_H
from adonis.constants import MASS_PDG_PROTON
from adonis.reweight.reduced_amps2 import build_res_reduced
from adonis.reweight.bank_reweight import _FSI_F

_MP = float(MASS_PDG_PROTON)


def build_chunk(n, seed):
    k_nu, k_lep, p_N, p_pi, w = (np.asarray(x) for x in generate_H(n, seed))
    p_struck = np.tile([_MP, 0.0, 0.0, 0.0], (n, 1))
    # ragged final state: 2 particles/event = [proton, pi+]; fs_off cumulative (2 per event)
    fs_p4 = np.empty((2 * n, 4), np.float32); fs_p4[0::2] = p_N; fs_p4[1::2] = p_pi
    fs_pid = np.tile(np.array([2212, 211], np.int32), n)
    fs_chg = np.tile(np.array([1, 1], np.int8), n)                 # proton +1, pi+ +1
    fs_off = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    rec = build_res_reduced(k_nu, k_lep, p_struck, p_N, p_pi,
                            np.full(n, 2212, np.int32), np.full(n, 211, np.int32))
    save = dict(
        w0=w.astype(np.float64), channel=np.ones(n, np.int8),
        k_nu=k_nu.astype(np.float32), k_mu=k_lep.astype(np.float32),
        p_struck=p_struck.astype(np.float32),
        fs_pid=fs_pid, fs_chg=fs_chg, fs_p4=fs_p4, fs_off=fs_off,
        res_p_N=p_N.astype(np.float32), res_p_pi=p_pi.astype(np.float32),
        res_ipid=np.full(n, 2212, np.int32), res_ppid=np.full(n, 211, np.int32),
        hv_res_mij=np.asarray(rec["M"], np.float64), hv_res_Q2=np.asarray(rec["Q2"], np.float32),
        hv_qe_mij=np.zeros((n, 4, 4), np.float32), hv_qe_Q2=np.zeros(n, np.float32),
    )
    for fld in _FSI_F:                                             # empty FSI -> reweight 1 (free H, no cascade)
        save[f"f_{fld}"] = np.zeros(0, np.int64 if fld in ("p_eidx", "n_eidx") else np.float32)
    return save, float(w.sum())


def main():
    out, n_per, n_chunks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.makedirs(out, exist_ok=True)
    sig = 0.0
    for c in range(n_chunks):
        save, s = build_chunk(n_per, seed0 + c)
        np.savez(os.path.join(out, f"chunk_{c:03d}.npz"), **save)
        sig += s
        print(f"chunk {c + 1}/{n_chunks}: {n_per} ev, sigma={s:.4e} nb", flush=True)
    json.dump(dict(n_chunks=n_chunks, chunk=n_per, n_total=n_per * n_chunks,
                   material="H", channels=["res"], probe="CC", note="free-proton CC1pi+ (no FSI/SF)"),
              open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"done: {n_chunks} chunks -> {out}  (mean sigma ~ {sig / n_chunks:.4e} nb)", flush=True)


if __name__ == "__main__":
    main()
