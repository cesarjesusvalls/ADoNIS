"""run_generation(GenConfig): the generation driver -- gen_cc_engine_rich logic, importable.

Per-event logic is lifted VERBATIM from scripts/gen_cc_engine_rich.py so banks stay bit-identical:
RES via res_xsec.generate / QE via qe_xsec.sample_importance (w/=N), the faithful BFS cascade
(cascade_full.cascade_nucleus, internal seed=1, key PRNGKey(seed+11)), top-M proton terminals with
provenance, per-seed checkpoint, weight normalized by the ACTUAL seeds banked.

Material: only a carbon cascade target is engine-generatable (free-H is a separate primary bank, not
wired); resolve_targets enforces the supported set.
"""
from __future__ import annotations
import os
import numpy as np
import jax
import jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.xsec.spectral import SpectralFunction
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi.cascade_real import _load_density
import adonis.fsi.cascade_full as CF
import adonis.fsi.tracking as TK
from adonis.workflow.materials import resolve_targets

_CHAN_OUT = {"res": "cc1pi", "qe": "cc0pi"}


def gen_events(channel, n, seed, sf_n=None, sf_p=None, n_neutron=6, n_proton=6, grid=None):
    """Primary events for one seed (verbatim gen_cc_engine_rich.gen_events).  sf_n/sf_p = the target's
    neutron/proton SpectralFunction (None -> carbon default inside the generator).  CC QE struck
    nucleon is a neutron (n->p) so it uses sf_n; RES uses both.  n_neutron/n_proton (target A-Z / Z)
    scale the initwgt = N*S normalization per species (default 6/6 = carbon)."""
    if channel == "res":
        e = res_xsec.generate(n, seed=seed, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=n_neutron, n_proton=n_proton, grid=grid)["events"]
        return {k: np.asarray(e[k]) for k in
                ("k_nu", "k_mu", "p_struck", "p_pi", "p_N", "w", "ppid", "ipid", "Npid")}
    from adonis.xsec import qe_xsec
    r = qe_xsec.sample_importance(n, seed=seed, sf=sf_n, n_neutron=n_neutron); m = len(r["w"])
    return dict(k_nu=np.asarray(r["k_nu"]), k_mu=np.asarray(r["k_mu"]), p_struck=np.asarray(r["p_struck"]),
                p_N=np.asarray(r["p_out"]), w=np.asarray(r["w"]) / n,
                ipid=np.full(m, 2112, np.int64), Npid=np.full(m, 2212, np.int64), ppid=np.zeros(m, np.int64))


def _prefsi_record(channel, a):
    """PRE-FSI record (engine schema) built from the primary interaction products -- no cascade.
    The recoil proton is the single primary nucleon (zeroed where it is a neutron, e.g. n->n pi+, so
    it is not counted as a proton); the primary pion is the RES pion (none for QE); no created pion."""
    nn = len(a["w"])
    Npid = np.asarray(a["Npid"]); is_p = (Npid == 2212)
    prot = np.where(is_p[:, None], np.asarray(a["p_N"]), 0.0)[:, None, :]          # (nn,1,4)
    pi4 = np.asarray(a["p_pi"]) if "p_pi" in a else np.zeros((nn, 4))
    pid_pi = np.asarray(a["ppid"]) if channel == "res" else np.zeros(nn, np.int64)
    z = np.zeros(nn)
    return dict(mu=a["k_mu"], nu=a["k_nu"], struck=a["p_struck"], pid_Ni=np.asarray(a["ipid"]).astype(np.int64),
                pi_post=pi4, pid_pi=pid_pi.astype(np.int64), pi_nsc=z.astype(np.int64),
                cr_p4=np.zeros((nn, 4)), cr_pid=np.zeros(nn, np.int64),
                prot=prot, prot_origin=np.where(is_p, 0, -1)[:, None].astype(np.int64),
                prot_gen=np.zeros((nn, 1), np.int64),
                w=np.asarray(a["w"]), ipid=np.asarray(a["ipid"]).astype(np.int64), Npid=Npid.astype(np.int64))


def run_one_seed(channel, n, seed, cas, cfg_cascade, track=False, fsi=True, sf_n=None, sf_p=None,
                 n_neutron=6, n_proton=6, grid=None, n_w=None):
    """One seed -> (record dict, truth dict|None, overflow).  Verbatim gen_cc_engine_rich.one (fsi=True);
    fsi=False returns the PRE-FSI primary record (no cascade)."""
    a = gen_events(channel, n, seed, sf_n=sf_n, sf_p=sf_p, n_neutron=n_neutron, n_proton=n_proton, grid=grid)
    if not fsi:
        return _prefsi_record(channel, a), None, 0
    nn = len(a["w"]); ar = np.arange(nn)
    p_pi_in = jnp.asarray(a["p_pi"]) if "p_pi" in a else jnp.asarray(a["p_N"])
    pterm, nterms, ofl, created = CF.cascade_nucleus(
        p_pi_in, jnp.asarray(a["p_N"]), jnp.asarray(a["ppid"], jnp.int32), jnp.asarray(a["ipid"], jnp.int32),
        jnp.asarray(a["Npid"], jnp.int32), cfg_cascade, jax.random.PRNGKey(seed + 11),
        P=cas.P, channel=channel, n_w=n_w)
    p4s, orgs, gns = [], [], []
    for g in nterms:
        sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
        good = (sp == CF.NUCLEON) & (pid == 2212) & al
        p4s.append(np.where(good[:, :, None], p4, 0.0))
        orgs.append(np.where(good, np.asarray(g["origin"]), -1))
        gns.append(np.where(good, np.asarray(g["gen"]), -1))
    P4 = np.concatenate(p4s, axis=1); ORG = np.concatenate(orgs, axis=1); GN = np.concatenate(gns, axis=1)
    mom = np.linalg.norm(P4[:, :, 1:], axis=2)
    idx = np.argsort(-mom, axis=1)[:, :cas.mprot]; g2 = ar[:, None]
    rec = dict(mu=a["k_mu"], nu=a["k_nu"], struck=a["p_struck"], pid_Ni=a["ipid"].astype(np.int64),
               pi_post=np.asarray(pterm["p4"]), pid_pi=np.asarray(pterm["pid"]).astype(np.int64),
               pi_nsc=np.asarray(pterm["nsc"]).astype(np.int64),
               cr_p4=np.asarray(created["p4"]), cr_pid=np.where(np.asarray(created["alive"]),
                                                               np.asarray(created["pid"]), 0).astype(np.int64),
               prot=P4[g2, idx], prot_origin=ORG[g2, idx].astype(np.int64), prot_gen=GN[g2, idx].astype(np.int64),
               w=a["w"], ipid=a["ipid"].astype(np.int64), Npid=a["Npid"].astype(np.int64))
    truth = None
    if track:
        truth = TK.finalize(nterms, cfg=TK.TrackerConfig(track=True))   # max_tracks: TrackerConfig default (64)
    return rec, truth, int(ofl)


def run_channel(channel, gc, target, n_w=None):
    """Generate one channel's bank (+ optional truth sidecar); per-seed checkpoint.  Returns out path.
    n_w forwards to the cascade refill engine (None=engine default; 0=full-batch lock-step).
    target = the resolved NuclearTarget (carbon/argon/...): its density_p/density_n/configs go into the
    cascade config and its spectral_n/spectral_p into the primary generators (single source of truth)."""
    cas = gc.cascade
    # Nucleus-aware cascade length: a nucleon must be able to traverse the FULL nucleus (diameter =
    # 2*radius, the rho<1e-6 escape cutoff) plus margin for scattered paths, else slow nucleons get
    # truncated mid-flight ("alive inside" -> spurious soft final state instead of escape/capture).  The
    # fixed 260 was carbon-tuned (C radius 6.55 fm) and far too short for larger nuclei (Ar 9.10 fm).
    # Scale ~3*radius/step; keep the YAML value as a floor.  early_exit makes max_steps a CAP -> the walk
    # stops once every nucleon has escaped, so a generous cap costs ~the actual escape distribution.
    _, _, _, _radius = _load_density(target.density_p, target.density_n)
    _ms = max(cas.max_steps, int(np.ceil(3.0 * _radius / cas.step)))
    print(f"[cascade] nucleus radius={_radius:.2f} fm -> max_steps={_ms} (floor {cas.max_steps})", flush=True)
    cfg_cascade = DiscreteCascadeConfig(step=cas.step, max_steps=_ms,
                                        seed=1, nn_inelastic=cas.nn_inelastic,
                                        nucleus=target.density_p, density_n=target.density_n,
                                        configs=target.configs)
    sf_n = SpectralFunction(target.spectral_n); sf_p = SpectralFunction(target.spectral_p)
    n_neutron = target.A - target.Z; n_proton = target.Z      # initwgt = N_species * S scaling
    out = os.path.join(gc.out_dir, f"t2k_{_CHAN_OUT[channel]}_engine_rich{gc.tag}.npz")
    truth_out = out.replace(".npz", "_truth.npz")
    # Optional frozen VegasGrid (RES importance estimator only) over the 6 final-state hypercube dims
    # (beam + 3-body).  Default OFF -> `grid=None` -> bit-identical sampling.  QE has no grid (its
    # struck-nucleon sampler is near-optimal).  The warm-up is ONE-TIME per material: the frozen grid is
    # a reproducible sidecar, so `cache="auto"` reuses it across runs (warm-up is deterministic anyway).
    grid = None
    vg = getattr(gc, "vegas", None)
    if vg is not None and vg.enabled and channel == "res":
        from adonis.xsec import res_xsec as _R
        from adonis.xsec.vegas_grid import VegasGrid
        # Grid cache key = the PHYSICS that determines the proposal: material (spectral fns + N counts)
        # and flux mode.  Flux is T2K-neutrino only today, so the canonical key is the material -> one
        # shared grid auto-reused across tags/seed-counts of the same nucleus.  (When a nubar/other flux
        # is wired, extend this key with the flux mode -- the grid's beam axis + lepton current change.)
        gp = vg.grid_path or os.path.join(gc.out_dir, f"res_vegasgrid_{target.symbol}{target.A}.npz")
        have = os.path.exists(gp)
        if vg.cache == "load" and not have:
            raise FileNotFoundError(f"vegas.cache='load' but no grid at {gp} (build one first / use 'auto').")
        if vg.cache != "rebuild" and have:                    # auto+present or load -> reuse cached grid
            grid = VegasGrid.load(gp)
            print(f"[{channel}] VEGAS grid loaded from cache -> {gp}", flush=True)
        else:                                                 # rebuild, or auto+missing -> warm up + save
            print(f"[{channel}] VEGAS warm-up: {vg.warmup_iters} iters x {vg.warmup_n} ev, nbins={vg.nbins}", flush=True)
            grid = _R.warmup_vegas(n=vg.warmup_n, iters=vg.warmup_iters, nbins=vg.nbins, alpha=vg.alpha,
                                   seed=vg.seed, sf_n=sf_n, sf_p=sf_p, n_neutron=n_neutron, n_proton=n_proton)
            grid.save(gp)
            print(f"[{channel}] VEGAS grid frozen -> {gp}", flush=True)
    parts, truths = [], []
    print(f"[{channel}] target={target.symbol}{target.A} Z={target.Z} N={n_neutron} "
          f"dens=({target.density_p},{target.density_n}) cfg={target.configs}", flush=True)
    print(f"[{channel}] {'PRE-FSI (no cascade)' if not gc.fsi else 'buffers: P=%d MPROT=%d' % (cas.P, cas.mprot)} "
          f"track={gc.tracking.enabled} -> {out}", flush=True)
    for k in range(gc.n_seeds):
        sd = gc.seed0 + k
        rec, truth, ofl = run_one_seed(channel, gc.n_per_seed, sd, cas, cfg_cascade, gc.tracking.enabled,
                                       gc.fsi, sf_n=sf_n, sf_p=sf_p, n_neutron=n_neutron, n_proton=n_proton,
                                       grid=grid, n_w=n_w)
        parts.append(rec)
        bank = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        bank["w"] = bank["w"] / len(parts)               # normalize by ACTUAL seeds banked
        np.savez(out, **bank)
        if gc.tracking.enabled:
            truths.append(truth)
            tb = {key: np.concatenate([t[key] for t in truths]) for key in truths[0]}
            np.savez(truth_out, **tb)
        print(f"  [{channel}] seed {k+1}/{gc.n_seeds}  banked {len(bank['w'])} ev  overflow {ofl}", flush=True)
    return out


def run_generation(gc, n_w=None):
    """Drive generation from a GenConfig.  Engine generates the carbon cascade target only.
    n_w forwards to the cascade refill engine (None=engine default; 0=full-batch lock-step)."""
    if gc.tracking.steps:
        raise NotImplementedError("per-step trajectory storage is not supported by the pool engine "
                                  "(it was a BFS-only capability); use the MC-truth track tree "
                                  "(adonis.fsi.tracking.finalize/print_tree) for cascade introspection.")
    targets = resolve_targets(gc.material)
    cascade_targets = [t for t, _ in targets if t.runs_cascade]
    if len(cascade_targets) != 1:
        raise NotImplementedError(
            f"engine generation supports a SINGLE cascade target; material {gc.material!r} resolved to "
            f"{[t.symbol for t,_ in targets]} (free-H is a separate primary bank, not wired).")
    target = cascade_targets[0]                       # carbon, argon, ... -- any registered nucleus
    return {ch: run_channel(ch, gc, target, n_w=n_w) for ch in gc.channels}
