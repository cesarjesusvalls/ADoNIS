"""TAGGED-BEAM cascade banks: pi+ / proton / neutron on 12C, in ACHILLES's CrossSection-mode geometry.

WHY.  The T2K sample cannot measure several FSI knobs, for two different reasons (docs/paper_plan.md
section 3): some are DEGENERATE against the hard vertex / spectral function, and some are INVISIBLE
because their channel is kinematically shut at T2K energies.  A tagged hadron beam fixes both:
  * pure FSI -- no hard vertex, no spectral function, so the FSI knobs face NO competing knobs;
  * the BEAM ENERGY is a free lever, so a channel that is closed at T2K can simply be opened:
      pi+ above |p| ~ 680 MeV/c  -> piN -> eta N' opens          -> s_conv becomes reachable
      p/n above KE ~ 290 MeV     -> NN -> N Delta -> NN pi opens -> s_NN_inelastic becomes reachable
    (at T2K, sigma_inel/sigma_tot is 1-3% and s_conv fires in 430 of 624,652 pion hits.)

GEOMETRY -- ACHILLES RunCascade.cc::InitCrossSection, exactly (the same setup the validated pi+12C
transparency figure used): the beam particle is fired along +z at an impact parameter b uniform in a
disk of radius R_DISK (card Cascade/Params/radius = 10 fm), starting at z = -1.05 * R_nuc (5% outside
the surface, status external_test -> no formation zone).  |p| is drawn uniform in [P_LO, P_HI] and
sigma(p) is extracted by BINNING the tagged incoming |p| -- not a per-point scan.

NORMALIZATION.  With b uniform in the disk, sigma(bin) = pi R^2 * <reacted> over the events in that bin.
The beam (p, b) distribution is theta-INDEPENDENT, so the per-bin denominator n_tried is a constant and
only the numerator is reweighted:
    sigma_X(bin, theta) = pi R^2 * sum_{i in bin} X_i * w_i(theta) / n_tried(bin),   X in {reacted, ...}
w_i(theta) = pool_fsi_reweight(record_i, theta).  This is EXACT and it is only meaningful because the
pion reweight now carries the SURVIVAL (mean-free-path) factor: with the old branch-only reweight a
common rescale of the pion sigmas left w == 1, so sigma_reaction could not respond to sabs at all.

REACTION COUNTING is post-Pauli, as in ACHILLES (Cascade.cc:903 records the vertex only inside `if(hit)`):
we use the per-particle nsc (realized interactions), NEVER the kind-1 `hh`/`nh` (which is the pre-Pauli
sampled hit and would over-count Pauli-blocked interactions).

Usage:
    python -m analysis.beams.beam_bank pip  --n 500000 --pmin 50   --pmax 1000 --out output/beam_pip_C
    python -m analysis.beams.beam_bank prot --n 500000 --pmin 300  --pmax 1400 --out output/beam_prot_C
    python -m analysis.beams.beam_bank neut --n 500000 --pmin 300  --pmax 1400 --out output/beam_neut_C
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

R_DISK = 10.0                                  # beam impact-parameter disk [fm] (card Params/radius)
PIR2_MB = np.pi * R_DISK ** 2 * 10.0           # pi R^2 in mb (1 fm^2 = 10 mb) = 3141.6 mb

# beam key -> (PID, cascade species, charge)   charge: pion index {0:pi+,1:pi0,2:pi-}; nucleon 1=p, 0=n
BEAMS = {"pip": (211, "PION", 0), "prot": (2212, "NUCLEON", 1), "neut": (2112, "NUCLEON", 0)}


def build(beam="pip", n=500_000, pmin=50.0, pmax=1000.0, seed=0, target="C",
          outdir="output/beam_bank", chunk=250_000, log=print, pauli=True):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.workflow.materials import resolve_targets
    from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density, sample_nucleons, _CH_MASS
    from adonis.fsi import cascade_full as CF
    from adonis.fsi.cascade_real import _MP_PHYS, _MN_PHYS

    pid, species, charge = BEAMS[beam]
    tg = resolve_targets(target)[0][0]
    cfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                step=0.04, pauli=pauli, nn_inelastic=True, engine="pool",
                                beam_zplane=True)  # CrossSection external_test beam -> z-plane escape (Deviation 2)
    _rg, _rp, _rn, radius = _load_density(cfg.nucleus, cfg.density_n)
    z0 = -1.05 * float(radius)                  # InitCrossSection: 5% outside the nuclear surface
    mass = float(_CH_MASS[charge]) if species == "PION" else (_MP_PHYS if charge == 1 else _MN_PHYS)
    # NB the beam particle is built ON-SHELL with the mass the cascade itself uses, so E^2 - m^2 = |p|^2
    # exactly.  (The soft-pion NaN of docs/logbook/pion_soft_sigma_nan.md comes from the RES generator
    # building the pion with the mpi0 KINEMATIC mass while the cascade uses the physical one -- a beam
    # pion is immune, verified down to |p| = 5 MeV/c.)
    os.makedirs(outdir, exist_ok=True)
    n_chunks = int(np.ceil(n / chunk))
    t0 = time.time()
    log(f"beam_bank[{beam}] pid={pid} {species} q={charge} m={mass:.3f} | |p| in [{pmin},{pmax}] MeV/c "
        f"| N={n:,} in {n_chunks} chunk(s) | R_disk={R_DISK} fm (pi R^2 = {PIR2_MB:.1f} mb) -> {outdir}")

    for c in range(n_chunks):
        m = min(chunk, n - c * chunk)
        key = jax.random.PRNGKey(seed + 1000 * c)
        k_mom, k_disk, k_nuc, k_run = jax.random.split(key, 4)
        mom = jax.random.uniform(k_mom, (m,), minval=pmin, maxval=pmax)          # TAGGED beam momentum
        u = jax.random.uniform(k_disk, (m, 2))
        br = R_DISK * jnp.sqrt(u[:, 0]); th = 2 * jnp.pi * u[:, 1]               # uniform in the disk
        E = jnp.sqrt(mom ** 2 + mass ** 2)
        p4 = jnp.stack([E, jnp.zeros(m), jnp.zeros(m), mom], axis=1)             # along +z
        pos0 = jnp.stack([br * jnp.cos(th), br * jnp.sin(th), jnp.full((m,), z0)], axis=1)

        kn, kp = jax.random.split(k_nuc, 2)
        npos, nmom, nisp = sample_nucleons(kn, m, cfg)
        A = nisp.shape[1]
        su = dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=jnp.zeros((m, A), bool),
                  ch0=jnp.full(m, charge if species == "PION" else 0, jnp.int32), kp=kp)
        _k1, knuc, _k2 = jax.random.split(su["kp"], 3)

        g0 = CF.empty_batch(m, 1)
        g0["alive"] = jnp.ones((m, 1), bool)
        g0["species"] = jnp.full((m, 1), CF.PION if species == "PION" else CF.NUCLEON, jnp.int32)
        g0["charge"] = jnp.full((m, 1), charge, jnp.int32)
        g0["p4"] = p4[:, None, :]; g0["pos"] = pos0[:, None, :]
        g0["origin"] = jnp.full((m, 1), CF._ORIG_PRIM_PI, jnp.int32)   # provenance tag for THE BEAM particle
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(m))
        g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]

        stepper = CF.make_pool_stepper(su, cfg, with_rec=True)
        out, _sofl, _oofl, prim_fate, (rec, rofl) = CF.run_cascade_pool(
            g0, stepper, knuc, su["consumed0"], M=12, max_steps=2000, M_out=24,
            prim_origin=CF._ORIG_PRIM_PI, rec_caps=(96, 256))
        assert int(rofl) == 0, f"FSI record overflow in chunk {c}"
        log(f"  chunk {c+1}/{n_chunks}: cascade done ({m:,} ev, {time.time()-t0:.0f}s)")

        O = {k: np.asarray(v) for k, v in out.items()}
        prim_fate = np.asarray(prim_fate)
        is_prim = (O["origin"] == CF._ORIG_PRIM_PI)
        # charge==3 tags an ETA meson (propagated after piN->etaN conversion; routed through the PION
        # species slot).  A surviving eta is NOT a pion -> excluded from pi_alive so an eta that escapes
        # counts as "no surviving pion" (absorbed), while one that back-converts regenerates a real pion.
        pi_alive = (O["species"] == CF.PION) & (O["charge"] != 3) & O["alive"]

        # ---- REACTION (post-Pauli), mirroring ACHILLES `if(hit)` --------------------------------- #
        if species == "PION":
            prim_pi = is_prim & pi_alive
            nsc_prim = (O["nsc"] * prim_pi).max(axis=1)                  # surviving primary pion scatters
            reacted = (prim_fate == CF.FATE_ABSORB) | (prim_fate == CF.FATE_CONVERT) | (nsc_prim > 0)
            has_pi = pi_alive.any(axis=1)
            absorbed = reacted & ~has_pi                                 # reacted, no surviving pion
        else:
            prim_N = is_prim & (O["species"] == CF.NUCLEON)
            nsc_prim = (O["nsc"] * prim_N).max(axis=1)                   # POST-PAULI nucleon interactions
            reacted = nsc_prim > 0
            absorbed = np.zeros(m, bool)                                 # nucleons are not absorbed
        # pion PRODUCTION by a nucleon beam = the NN->NNpi inelastic channel -> the s_NN_inelastic handle
        n_pi_out = pi_alive.sum(axis=1).astype(np.int16)
        n_p_out = ((O["species"] == CF.NUCLEON) & O["alive"] & (O["charge"] == 1)).sum(axis=1).astype(np.int16)
        n_n_out = ((O["species"] == CF.NUCLEON) & O["alive"] & (O["charge"] == 0)).sum(axis=1).astype(np.int16)

        flat = CF.compact_fsi_record({k: np.asarray(v) for k, v in rec.items()})
        fsi_save = {"f_p_eidx": flat["p_eidx"].astype(np.int32), "f_n_eidx": flat["n_eidx"].astype(np.int32)}
        for f, arr in flat.items():
            if f.endswith("_eidx"):
                continue
            dt = (np.int8 if f in ("bc", "iso") else
                  (bool if f in ("hh", "inel", "swap", "pi_hh") else np.float32))
            fsi_save[f"f_{f}"] = arr.astype(dt)

        np.savez(f"{outdir}/chunk_{c:03d}.npz",
                 beam_p=np.asarray(mom, np.float32),
                 w0=np.full(m, PIR2_MB, np.float64),        # pi R^2 [mb]; sigma(bin) = <w0 * X * w(theta)>
                 reacted=reacted, absorbed=absorbed,
                 n_pi_out=n_pi_out, n_p_out=n_p_out, n_n_out=n_n_out,
                 nsc_prim=np.asarray(nsc_prim, np.int16), prim_fate=prim_fate.astype(np.int8),
                 **fsi_save)
        log(f"  chunk {c+1}/{n_chunks}: written | reacted {reacted.mean():.3f} | absorbed {absorbed.mean():.3f} "
            f"| <n_pi_out> {n_pi_out.mean():.3f} | pion slots {len(flat['p_eidx']):,} nucleon {len(flat['n_eidx']):,}")

    json.dump(dict(beam=beam, pid=pid, species=species, charge=charge, pmin=pmin, pmax=pmax,
                   n_total=n, n_chunks=n_chunks, R_disk=R_DISK, pir2_mb=PIR2_MB, target=target, seed=seed),
              open(f"{outdir}/manifest.json", "w"), indent=1)
    log(f"DONE {outdir} in {time.time()-t0:.0f}s")


def load(outdir):
    """Concatenate chunks; the ragged FSI slot indices are OFFSET into the global event numbering."""
    import glob
    man = json.load(open(f"{outdir}/manifest.json"))
    files = sorted(glob.glob(f"{outdir}/chunk_*.npz"))
    per = {}
    ev_off = 0
    for f in files:
        d = np.load(f)
        nev = len(d["w0"])
        for k in d.files:
            v = d[k]
            if k in ("f_p_eidx", "f_n_eidx"):
                v = v.astype(np.int64) + ev_off       # per-slot event index -> global numbering
            per.setdefault(k, []).append(v)
        ev_off += nev
    B = {k: np.concatenate(v) for k, v in per.items()}
    B["manifest"] = man
    return B


def beam_weight(B, knobs):
    """w(theta) for a cascade-only bank: w0 x the kind-1 FSI reweight (NO hard vertex, NO spectral fn)."""
    import jax.numpy as jnp
    from adonis.fsi.cascade_full import pool_fsi_reweight
    from analysis.t2k.differentiability.bank_reweight import _FSI_F
    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in _FSI_F}
    rec["n_events"] = len(B["w0"])
    k = knobs
    fsi = pool_fsi_reweight(rec, k["sabs"], 1.0, s_piN_elastic=k["s_piN_elastic"], s_piN_cex=k["s_piN_cex"],
                            s_conv=k["s_conv"], s_NN_elastic=k["s_NN_elastic"],
                            s_NN_inelastic=k["s_NN_inelastic"], f_NN_cex=k["f_NN_cex"])
    return jnp.asarray(B["w0"]) * fsi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("beam", choices=sorted(BEAMS))
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--pmin", type=float, required=True)
    ap.add_argument("--pmax", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=50_000,
                    help="events per chunk.  The DENSE record buffers (n x 96 pion + n x 256 nucleon "
                         "slots) are allocated per chunk BEFORE the ragged compaction, so peak memory "
                         "scales with THIS, not with the bank size.  250k thrashed a 16 GB box "
                         "(6.6 GB RSS, 8.2/9.2 GB swap); 50k is comfortable.")
    ap.add_argument("--no-pauli", action="store_true", help="DEBUG: disable Pauli blocking (ablation)")
    a = ap.parse_args()
    out = a.out or f"output/beam_{a.beam}_C"
    t0 = time.time()
    build(a.beam, n=a.n, pmin=a.pmin, pmax=a.pmax, seed=a.seed, outdir=out, chunk=a.chunk,
          pauli=not a.no_pauli,
          log=lambda s: print(f"[{time.time()-t0:7.1f}s] {s}", flush=True))
