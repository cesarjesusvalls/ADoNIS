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

Usage (config-driven, mirrors adonis.workflow.cli):
    python -m analysis.paper.beams.beam_bank configs/beam_pip_C.yaml
    python -m analysis.paper.beams.beam_bank configs/beam_prot_C.yaml
    python -m analysis.paper.beams.beam_bank configs/beam_neut_C.yaml
Any config field is overridable on the CLI (--n, --pmin, --pmax, --chunk, --seed, --tag, --out, --no-pauli).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from adonis.flux.hadron import BEAMS, R_DISK, PIR2_MB, HadronBeam    # beam description (adonis/flux)


def build(beam="pip", n=500_000, pmin=50.0, pmax=1000.0, seed=0, target="C",
          outdir="output/beam_bank", chunk=250_000, log=print, pauli=True):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.workflow.materials import resolve_targets
    from adonis.fsi.cascade import DiscreteCascadeConfig, _load_density, sample_nucleons, _CH_MASS
    from adonis.fsi import cascade as CF
    from adonis.fsi.cascade import _MP_PHYS, _MN_PHYS

    pid, species, charge = BEAMS[beam]
    tg = resolve_targets(target)[0][0]
    cfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                step=0.04, pauli=pauli, nn_inelastic=True, engine="pool")
    _rg, _rp, _rn, radius = _load_density(cfg.nucleus, cfg.density_n)
    z0 = -1.05 * float(radius)                  # InitCrossSection: 5% outside the nuclear surface
    mass = float(_CH_MASS[charge]) if species == "PION" else (_MP_PHYS if charge == 1 else _MN_PHYS)
    hb = HadronBeam(beam, mass)                 # reusable beam description (adonis/flux/hadron)
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
        mom, p4, pos0 = hb.sample(k_mom, k_disk, m, pmin, pmax, z0)              # beam (adonis/flux/hadron)

        kn, kp = jax.random.split(k_nuc, 2)
        npos, nmom, nisp = sample_nucleons(kn, m, cfg)
        A = nisp.shape[1]
        # FSI-record slot caps must SCALE WITH THE NUCLEUS: a heavier target gives longer cascades and
        # more vertices per event.  Overflow is a hard assert below, so under-sizing kills the shard --
        # (96,256) is validated for 12C but overflowed on 40Ar (pion beam worst: 112 shards lost).
        # ceil(A/12): 12C -> 1x (unchanged), 40Ar -> 4x.
        _sc = max(1, -(-A // 12))
        su = dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=jnp.zeros((m, A), bool),
                  ch0=jnp.full(m, charge if species == "PION" else 0, jnp.int32), kp=kp)
        _k1, knuc, _k2 = jax.random.split(su["kp"], 3)

        g0 = CF.empty_batch(m, 1)
        g0["alive"] = jnp.ones((m, 1), bool)
        g0["species"] = jnp.full((m, 1), CF.PION if species == "PION" else CF.NUCLEON, jnp.int32)
        g0["charge"] = jnp.full((m, 1), charge, jnp.int32)
        g0["p4"] = p4[:, None, :]; g0["pos"] = pos0[:, None, :]
        g0["origin"] = jnp.full((m, 1), CF._ORIG_PRIM_PI, jnp.int32)   # provenance tag for THE BEAM particle
        g0["external_test"] = jnp.ones((m, 1), bool)   # ACHILLES external_test: CrossSection beam -> z-plane escape
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(m))
        g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]

        stepper = CF.make_pool_stepper(su, cfg, with_rec=True)
        out, _sofl, _oofl, prim_fate, (rec, rofl) = CF.run_cascade_pool(
            g0, stepper, knuc, su["consumed0"], M=12, max_steps=2000, M_out=24,
            prim_origin=CF._ORIG_PRIM_PI, rec_caps=(96 * _sc, 256 * _sc))
        assert int(rofl) == 0, f"FSI record overflow in chunk {c}"
        log(f"  chunk {c+1}/{n_chunks}: cascade done ({m:,} ev, {time.time()-t0:.0f}s)")

        O = {k: np.asarray(v) for k, v in out.items()}
        prim_fate = np.asarray(prim_fate)[:, 0]     # run_cascade_pool now returns per-primary (n,Wprim);
        #   a beam has one primary (col 0) -- keeps this reference generator byte-for-byte for validation
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
            # A primary nucleon that reacted and was then CAPTURED (KE<10 MeV) is bound -> removed from the
            # final-state output (D12), so nsc_prim can't see it (nsc_prim=0).  Its capture is latched in
            # prim_fate (FATE_CAPTURE); count it as reacted (a captured primary necessarily scattered to
            # slow below 10 MeV).  Without this, captured primaries were mis-counted as non-reacted ->
            # low-momentum reaction + N_p=0 deficit (v2->v2k regression).
            reacted = (nsc_prim > 0) | (prim_fate == CF.FATE_CAPTURE)
            absorbed = np.zeros(m, bool)                                 # nucleons are not absorbed
        # pion PRODUCTION by a nucleon beam = the NN->NNpi inelastic channel -> the s_NN_inelastic handle
        n_pi_out = pi_alive.sum(axis=1).astype(np.int16)
        n_p_out = ((O["species"] == CF.NUCLEON) & O["alive"] & (O["charge"] == 1)).sum(axis=1).astype(np.int16)
        n_n_out = ((O["species"] == CF.NUCLEON) & O["alive"] & (O["charge"] == 0)).sum(axis=1).astype(np.int16)
        # charge-resolved final-state multiplicities.  PION charge index: 0=pi+, 1=pi0, 2=pi-, 3=eta
        # (the eta rides the PION species slot).  Nucleon charge: 1=p, 0=n (already in n_p/n_n_out).
        _PI = (O["species"] == CF.PION) & O["alive"]
        n_pip = (_PI & (O["charge"] == 0)).sum(axis=1).astype(np.int16)
        n_pi0 = (_PI & (O["charge"] == 1)).sum(axis=1).astype(np.int16)
        n_pim = (_PI & (O["charge"] == 2)).sum(axis=1).astype(np.int16)
        n_eta = (_PI & (O["charge"] == 3)).sum(axis=1).astype(np.int16)

        # ---- FLATTENED per-particle final-state kinematics (every ESCAPED particle) ------------------ #
        # Ragged over the (m, M_out) output buffer: keep only alive slots, flatten, tag with event index
        # (offset into the global numbering in load(), like the FSI eidx).  Gives momentum + angular
        # spectra per species/charge without re-running.  cth = pz/|p| vs the +z beam axis.
        _al = O["alive"]
        _p3 = O["p4"][:, :, 1:]
        _pm = np.linalg.norm(_p3, axis=2)
        _cth = _p3[:, :, 2] / np.clip(_pm, 1e-9, None)
        _eidx_grid = np.broadcast_to(np.arange(m, dtype=np.int32)[:, None], _al.shape)
        ks_save = {
            "ks_eidx": _eidx_grid[_al].astype(np.int32),
            "ks_species": O["species"][_al].astype(np.int8),
            "ks_charge": O["charge"][_al].astype(np.int8),
            "ks_pmag": _pm[_al].astype(np.float32),
            "ks_cth": _cth[_al].astype(np.float32),
            "ks_prim": (O["origin"][_al] == CF._ORIG_PRIM_PI),
        }
        # GUARD (D12 invariant): an EMITTED nucleon must have escaped, i.e. KE >= recap_ke; a captured
        # (bound) nucleon is partitioned out at the source and never enters the final state.  This catches
        # any regression that re-emits bound nucleons -- e.g. the old set-to-rest bug put captured protons
        # in the output at |p|=0 (KE=0), deflating P(n_p=0).  Loose floor (recap_ke - 5) so a genuine
        # near-threshold escaper never trips it, but a rest/bound nucleon does.
        _isN = ks_save["ks_species"] == CF.NUCLEON
        if _isN.any():
            _mN = np.where(ks_save["ks_charge"][_isN] == 1, _MP_PHYS, _MN_PHYS).astype(np.float64)
            _keN = np.sqrt(_mN ** 2 + ks_save["ks_pmag"][_isN].astype(np.float64) ** 2) - _mN
            assert _keN.min() > cfg.recap_ke - 5.0, (
                f"emitted nucleon KE={_keN.min():.2f} MeV < floor {cfg.recap_ke-5.0} -- capture leak "
                f"(a bound nucleon reached the final state; check _nucleon_step escape/capture partition)")

        # ---- FULL final-state 4-vectors (fs_*), UNIFORM with the neutrino/(e,e') banks -------------- #
        # Same ragged layout as event_bank.compact_fs: fs_off (n+1 offsets), fs_pid/fs_chg/fs_p4 over the
        # ALIVE slots.  So one final-state extraction reads every bank identically.  pid: pion slot ->
        # {pi+,pi0,pi-,eta}[charge], nucleon -> p/n; eta (charge 3) rides the PION species slot.
        _p4o = O["p4"].astype(np.float64)
        _fpid = np.where(O["species"] == CF.PION,
                         np.array([211, 111, -211, 221])[np.clip(O["charge"], 0, 3)],
                         np.where(O["charge"] == 1, 2212, 2112))
        _fcnt = _al.sum(1).astype(np.int64); _foff = np.concatenate([[0], np.cumsum(_fcnt)])
        _fm = _al.reshape(-1)
        fs_save = {"fs_off": _foff.astype(np.int64),
                   "fs_pid": _fpid.reshape(-1)[_fm].astype(np.int32),
                   "fs_chg": O["charge"].reshape(-1)[_fm].astype(np.int32),
                   "fs_p4": _p4o.reshape(-1, 4)[_fm].astype(np.float32)}

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
                 n_pip=n_pip, n_pi0=n_pi0, n_pim=n_pim, n_eta=n_eta,
                 nsc_prim=np.asarray(nsc_prim, np.int16), prim_fate=prim_fate.astype(np.int8),
                 **fsi_save, **ks_save, **fs_save)
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
            if k in ("f_p_eidx", "f_n_eidx", "ks_eidx"):
                v = v.astype(np.int64) + ev_off       # per-slot event index -> global numbering
            per.setdefault(k, []).append(v)
        ev_off += nev
    B = {k: np.concatenate(v) for k, v in per.items()}
    B["manifest"] = man
    return B


def beam_weight(B, knobs):
    """w(theta) for a cascade-only bank: w0 x the kind-1 FSI reweight (NO hard vertex, NO spectral fn)."""
    import jax.numpy as jnp
    from adonis.fsi.cascade import pool_fsi_reweight
    from adonis.reweight.bank_reweight import _FSI_F
    rec = {f: jnp.asarray(B[f"f_{f}"]) for f in _FSI_F}
    rec["n_events"] = len(B["w0"])
    k = knobs
    fsi = pool_fsi_reweight(rec, k.sabs, 1.0, s_piN_elastic=k.s_piN_elastic, s_piN_cex=k.s_piN_cex,
                            s_conv=k.s_conv, s_NN_elastic=k.s_NN_elastic,
                            s_NN_inelastic=k.s_NN_inelastic, f_NN_cex=k.f_NN_cex)
    return jnp.asarray(B["w0"]) * fsi


if __name__ == "__main__":
    from adonis.workflow.config import load_beam_config
    ap = argparse.ArgumentParser(description="Tagged-beam cascade bank (config-driven; mirrors adonis.workflow.cli).")
    ap.add_argument("config", help="configs/beam_*.yaml (BeamGenConfig)")
    # per-run overrides onto the loaded config (all optional; None -> keep the config value)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pmin", type=float, default=None)
    ap.add_argument("--pmax", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out", default=None, help="override the bank dir (default <out_dir>/beam_<beam>_<material><tag>)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="events per chunk.  The DENSE record buffers (n x 96 pion + n x 256 nucleon "
                         "slots) are allocated per chunk BEFORE the ragged compaction, so peak memory "
                         "scales with THIS, not with the bank size.  250k thrashed a 16 GB box "
                         "(6.6 GB RSS, 8.2/9.2 GB swap); 50k is comfortable.")
    ap.add_argument("--no-pauli", action="store_true", help="DEBUG: disable Pauli blocking (ablation)")
    a = ap.parse_args()

    bc = load_beam_config(a.config)
    if a.n is not None:     bc.n = a.n
    if a.pmin is not None:  bc.pmin = a.pmin
    if a.pmax is not None:  bc.pmax = a.pmax
    if a.seed is not None:  bc.seed = a.seed
    if a.chunk is not None: bc.chunk = a.chunk
    if a.tag is not None:   bc.tag = a.tag
    if a.no_pauli:          bc.pauli = False
    bc.__post_init__()                                  # re-validate after overrides
    out = a.out or bc.bank_dir
    t0 = time.time()
    build(bc.beam, n=bc.n, pmin=bc.pmin, pmax=bc.pmax, seed=bc.seed, target=bc.material, outdir=out,
          chunk=bc.chunk, pauli=bc.pauli,
          log=lambda s: print(f"[{time.time()-t0:7.1f}s] {s}", flush=True))
