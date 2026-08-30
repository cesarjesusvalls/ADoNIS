"""Free-proton bank generation: nu_mu p -> mu- p pi+.

A free nucleon has no spectral function and no nuclear cascade, so it does not go through
generate_bank's chunk loop.  The bank it writes is schema-compatible with one that does: the ragged
final state, w0, the exact reduced-quadratic RES hard-vertex record, a zero QE record, and empty FSI
records, so the FSI reweight is identically one.

RES only: there is no CC quasi-elastic channel on a proton.

    python -m adonis.workflow.generate_h_bank configs/banks/nu_MINERvA_H.yaml --out output/nu_MINERvA_H \\
        --n-per-seed 1000000 --n-seeds 5

The flux comes from the config, like every other bank.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from adonis.workflow.config import load_gen_config, FLUX_FILES

_FSI_FIELDS = None


def build_chunk(n, seed, flux_file=None):
    """One chunk's record set, and its cross-section contribution [nb]."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from adonis.channels.free_proton import generate_H
    from adonis.constants import MASS_PDG_PROTON
    from adonis.reweight.reduced_amps2 import build_res_reduced
    from adonis.reweight.bank_reweight import _FSI_F

    m_p = float(MASS_PDG_PROTON)
    k_nu, k_lep, p_N, p_pi, w = (np.asarray(x) for x in generate_H(n, seed, flux=flux_file))
    p_struck = np.tile([m_p, 0.0, 0.0, 0.0], (n, 1))

    fs_p4 = np.empty((2 * n, 4), np.float32)
    fs_p4[0::2] = p_N
    fs_p4[1::2] = p_pi
    rec = build_res_reduced(k_nu, k_lep, p_struck, p_N, p_pi,
                            np.full(n, 2212, np.int32), np.full(n, 211, np.int32))
    save = dict(
        w0=w.astype(np.float64), channel=np.ones(n, np.int8),
        k_nu=k_nu.astype(np.float32), k_mu=k_lep.astype(np.float32),
        p_struck=p_struck.astype(np.float32),
        fs_pid=np.tile(np.array([2212, 211], np.int32), n),
        fs_chg=np.tile(np.array([1, 1], np.int8), n),
        fs_p4=fs_p4, fs_off=np.arange(0, 2 * n + 1, 2, dtype=np.int64),
        res_p_N=p_N.astype(np.float32), res_p_pi=p_pi.astype(np.float32),
        res_ipid=np.full(n, 2212, np.int32), res_ppid=np.full(n, 211, np.int32),
        hv_res_mij=np.asarray(rec["M"], np.float64), hv_res_Q2=np.asarray(rec["Q2"], np.float32),
        hv_qe_mij=np.zeros((n, 4, 4), np.float32), hv_qe_Q2=np.zeros(n, np.float32),
    )
    for fld in _FSI_F:
        save[f"f_{fld}"] = np.zeros(0, np.int64 if fld in ("p_eidx", "n_eidx") else np.float32)
    return save, float(w.sum())


def generate(cfg, outdir, log=print):
    """Write <outdir>/chunk_NNN.npz + manifest.json for a free-nucleon GenConfig."""
    flux_file = FLUX_FILES[cfg.flux]
    os.makedirs(outdir, exist_ok=True)
    log(f"[H bank] {cfg.material} free nucleon: flux={cfg.flux} ({flux_file})")

    sigma = 0.0
    for c in range(cfg.n_seeds):
        save, s = build_chunk(cfg.n_per_seed, cfg.seed0 + c, flux_file)
        np.savez(os.path.join(outdir, f"chunk_{c:03d}.npz"), **save)
        sigma += s
        log(f"  chunk {c + 1}/{cfg.n_seeds}: {cfg.n_per_seed} ev, sigma={s:.4e} nb")

    manifest = dict(n_chunks=cfg.n_seeds, chunk=cfg.n_per_seed,
                    n_total=cfg.n_per_seed * cfg.n_seeds, material=cfg.material,
                    channels=list(cfg.channels), probe=cfg.probe, flux=cfg.flux,
                    flux_file=flux_file, note="free proton, no spectral function and no cascade")
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"[H bank] {cfg.n_seeds} chunks -> {outdir}  (mean sigma {sigma / cfg.n_seeds:.4e} nb)")
    return outdir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adonis.workflow.generate_h_bank",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("config", help="a free-nucleon GenConfig YAML (material: H)")
    ap.add_argument("--out", default=None, help="bank output dir")
    ap.add_argument("--n-per-seed", type=int, default=None, help="events per chunk")
    ap.add_argument("--n-seeds", type=int, default=None, help="chunks this shard writes")
    ap.add_argument("--seed0", type=int, default=None, help="first seed")
    a = ap.parse_args(argv)

    cfg = load_gen_config(a.config)
    if a.n_per_seed is not None:
        cfg.n_per_seed = a.n_per_seed
    if a.n_seeds is not None:
        cfg.n_seeds = a.n_seeds
    if a.seed0 is not None:
        cfg.seed0 = a.seed0
    out = a.out or os.path.join(cfg.out_dir, f"{cfg.bank_prefix}_{cfg.material}{cfg.tag}")
    generate(cfg, out)
    print("DONE:", out, flush=True)


if __name__ == "__main__":
    main()
