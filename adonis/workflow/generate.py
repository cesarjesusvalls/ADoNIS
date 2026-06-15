"""run_generation(GenConfig): the generation driver -- gen_cc_engine_rich logic, importable.

Per-event logic is lifted VERBATIM from scripts/gen_cc_engine_rich.py so banks stay bit-identical:
RES via res_xsec.generate / QE via qe_xsec.sample_importance (w/=N), the faithful BFS cascade
(cascade_full.cascade_carbon_v2, internal seed=1, key PRNGKey(seed+11)), top-M proton terminals with
provenance, per-seed checkpoint, weight normalized by the ACTUAL seeds banked.

`_N_RECOIL` is read from ADONIS_N_RECOIL at cascade_discrete import time, so the driver
(scripts/adonis_generate.py) MUST set that env from cfg.cascade.n_recoil BEFORE importing this module;
we assert consistency here.  Material: only a carbon cascade target is engine-generatable (free-H is a
separate primary bank, not wired); resolve_targets enforces the supported set.
"""
from __future__ import annotations
import os
import numpy as np
import jax
import jax.numpy as jnp
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec import res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _N_RECOIL
import adonis.fsi.cascade_full as CF
import adonis.fsi.tracking as TK
from adonis.workflow.materials import resolve_targets

_CHAN_OUT = {"res": "cc1pi", "qe": "cc0pi"}


def gen_events(channel, n, seed):
    """Primary events for one seed (verbatim gen_cc_engine_rich.gen_events)."""
    if channel == "res":
        e = res_xsec.generate(n, seed=seed, return_events=True)["events"]
        return {k: np.asarray(e[k]) for k in
                ("k_nu", "k_mu", "p_struck", "p_pi", "p_N", "w", "ppid", "ipid", "Npid")}
    from adonis.xsec import qe_xsec
    r = qe_xsec.sample_importance(n, seed=seed); m = len(r["w"])
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


def run_one_seed(channel, n, seed, cas, cfg_cascade, track=False, fsi=True):
    """One seed -> (record dict, truth dict|None, overflow).  Verbatim gen_cc_engine_rich.one (fsi=True);
    fsi=False returns the PRE-FSI primary record (no cascade)."""
    a = gen_events(channel, n, seed)
    if not fsi:
        return _prefsi_record(channel, a), None, 0
    nn = len(a["w"]); ar = np.arange(nn)
    p_pi_in = jnp.asarray(a["p_pi"]) if "p_pi" in a else jnp.asarray(a["p_N"])
    pterm, nterms, ofl, created = CF.cascade_carbon_v2(
        p_pi_in, jnp.asarray(a["p_N"]), jnp.asarray(a["ppid"], jnp.int32), jnp.asarray(a["ipid"], jnp.int32),
        jnp.asarray(a["Npid"], jnp.int32), cfg_cascade, jax.random.PRNGKey(seed + 11),
        P=cas.P, max_gen=cas.max_gen, channel=channel)
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
        truth = TK.finalize(nterms, cfg=TK.TrackerConfig(track=True, max_tracks=cas.P * cas.max_gen + 1))
    return rec, truth, int(ofl)


def run_channel(channel, gc):
    """Generate one channel's bank (+ optional truth sidecar); per-seed checkpoint.  Returns out path."""
    cas = gc.cascade
    if _N_RECOIL != cas.n_recoil:
        raise RuntimeError(f"_N_RECOIL={_N_RECOIL} != cascade.n_recoil={cas.n_recoil}: set "
                           f"ADONIS_N_RECOIL before importing adonis.workflow.generate (driver does this).")
    cfg_cascade = DiscreteCascadeConfig(cylinder=cas.cylinder, step=cas.step, max_steps=cas.max_steps,
                                        seed=1, nn_inelastic=cas.nn_inelastic)
    out = os.path.join(gc.out_dir, f"t2k_{_CHAN_OUT[channel]}_engine_rich{gc.tag}.npz")
    truth_out = out.replace(".npz", "_truth.npz")
    parts, truths = [], []
    print(f"[{channel}] {'PRE-FSI (no cascade)' if not gc.fsi else 'buffers: P=%d max_gen=%d N_RECOIL=%d MPROT=%d' % (cas.P, cas.max_gen, _N_RECOIL, cas.mprot)} "
          f"track={gc.tracking.enabled} -> {out}", flush=True)
    for k in range(gc.n_seeds):
        sd = gc.seed0 + k
        rec, truth, ofl = run_one_seed(channel, gc.n_per_seed, sd, cas, cfg_cascade, gc.tracking.enabled, gc.fsi)
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


def run_generation(gc):
    """Drive generation from a GenConfig.  Engine generates the carbon cascade target only."""
    if gc.tracking.steps:
        raise NotImplementedError("per-step trajectory storage in batch generation is not supported; "
                                  "use scripts/cascade_viz.py for viz (small N).")
    targets = resolve_targets(gc.material)
    cascade_targets = [t for t, _ in targets if t.runs_cascade]
    if len(cascade_targets) != 1 or cascade_targets[0].symbol != "C":
        raise NotImplementedError(
            f"engine generation supports a single carbon cascade target; material {gc.material!r} "
            f"resolved to {[t.symbol for t,_ in targets]} (free-H is a separate primary bank, not wired).")
    return {ch: run_channel(ch, gc) for ch in gc.channels}
