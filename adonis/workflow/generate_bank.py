"""generate_bank(GenConfig): the bank generator for every probe.

The only probe-specific code is the primary interaction (weak/EM hard vertex vs a tagged hadron
projectile); everything downstream -- cascade engine, cascade-outcome record builder
(adonis.workflow.records), chunk/seed loop, save -- is shared, driven by the config fields.

Per chunk (seed = cfg.seed0 + c) -> chunk_NNN.npz + manifest.json. Uniform record set (all probes):
fs_* final state, f_* FSI kind-1, n_* multiplicities, ks_* escaped list, reacted/absorbed. Plus the
primary's own kinematics/weight (weak: k_nu/k_lep/hv_*; EM: c/omega/theta; hadron: beam_p/w0).
"""
from __future__ import annotations
import os, time, json, math, glob, gc as _gc
from pathlib import Path
import numpy as np

from adonis.workflow import records as REC




def _lepton_theta_deg(k_lep):
    """Outgoing-lepton polar angle [deg] about +z (the neutrino / e- beam axis)."""
    p3 = np.asarray(k_lep)[:, 1:]
    return np.degrees(np.arccos(np.clip(p3[:, 2] / np.clip(np.linalg.norm(p3, axis=1), 1e-9, None), -1.0, 1.0)))


def _accept_lepton(d, theta_acc, kkey=None, theta=None):
    """Uniform outgoing-lepton angular acceptance, shared by every hard-vertex channel (weak muon + EM
    electron).  Keeps events whose lepton polar angle is within [lo,hi] deg, masking every length-n
    field of the per-event dict `d`.  The angle is the precomputed `theta` (the EM channels already
    expose it) else computed from d[kkey] (the weak outgoing muon, k_lep).  Full acceptance (lo<=0 and
    hi>=180) short-circuits to the identity, so the weak default is byte-for-byte unaffected."""
    lo, hi = theta_acc
    if lo <= 0.0 and hi >= 180.0:
        return d
    th = np.asarray(theta) if theta is not None else _lepton_theta_deg(d[kkey])
    m = (th >= lo) & (th <= hi); n = len(th)
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in d.items()}


def _outdir(cfg):
    return os.path.join(cfg.out_dir, f"{cfg.bank_prefix}_{cfg.material}{cfg.tag}")


def generate_bank(cfg, outdir=None, log=None):
    outdir = outdir or _outdir(cfg)
    t0 = time.time()
    if log is None:
        def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    if cfg.probe == "hadron":
        return _generate_hadron(cfg, outdir, log, t0)
    from adonis.nuclear.targets import resolve_targets
    if resolve_targets(cfg.material)[0][0].free_nucleon:
        from adonis.workflow import generate_h_bank
        return generate_h_bank.generate(cfg, outdir, log)
    return _generate_hardvertex(cfg, outdir, log, t0)


def _generate_hardvertex(cfg, outdir, log, t0):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.nuclear.targets import resolve_targets
    from adonis.nuclear.spectral import SpectralFunction
    import adonis.fsi.cascade as CF

    EM = (cfg.probe == "EM")
    NC = (cfg.probe == "NC")
    CHUNK = cfg.chunk or cfg.n_per_seed
    n_chunks = cfg.n_seeds
    SEED0 = cfg.seed0
    do_qe = "qe" in cfg.channels
    do_res = "res" in cfg.channels
    if not EM:
        from adonis.flux.spectrum import set_default_flux
        from adonis.workflow.config import FLUX_FILES
        set_default_flux(FLUX_FILES[cfg.flux])
    tgt = resolve_targets(cfg.material)[0][0]
    n_neutron = tgt.A - tgt.Z; n_proton = tgt.Z
    _MARGIN = float(os.environ.get("ADONIS_REC_MARGIN", "1.5"))
    POOL = lambda **k: CF.pool_cascade_config(nucleus=tgt.density_p, density_n=tgt.density_n,
                                              configs=tgt.configs, **k)

    LACC = tuple(cfg.theta_acc)
    _ALL = (0.0, 180.0)
    if EM:
        from adonis.channels import ee as ee_x, res_ee as res_ee_x
        sf = SpectralFunction(tgt.spectral_n)
        EB = float(cfg.e_beam)

        def gen_qe(n, seed):
            r = ee_x.generate(n, material=cfg.material, seed=seed, E_beam=EB, records=True, theta_acc=_ALL)
            r = _accept_lepton(r, LACC, theta=r["theta"])
            pid = np.where(r["is_p"], 2212, 2112).astype(np.int32)
            return dict(c=np.asarray(r["c"]), omega=np.asarray(r["omega"]), theta=np.asarray(r["theta"]),
                        k_lep=np.asarray(r["k_lep"]),
                        p_N=np.asarray(r["p_out"]), p_pi=np.zeros((len(pid), 4)),
                        ppid=np.zeros(len(pid), np.int32), ipid=pid, Npid=pid, _raw=r)

        def gen_res(n, seed):
            r = res_ee_x.generate(n, material=cfg.material, seed=seed, E_beam=EB, records=True, theta_acc=_ALL)
            r = _accept_lepton(r, LACC, theta=r["theta"])
            return dict(c=np.asarray(r["c"]), omega=np.asarray(r["omega"]), theta=np.asarray(r["theta"]),
                        k_lep=np.asarray(r["k_lep"]),
                        p_N=np.asarray(r["p_N"]), p_pi=np.asarray(r["p_pi"]),
                        ppid=np.asarray(r["ppid"], np.int32), ipid=np.asarray(r["ipid"], np.int32),
                        Npid=np.asarray(r["Npid"], np.int32), _raw=r)
        m_extra = dict(E_beam=EB, theta_acc=list(LACC))
    elif NC:
        from adonis.channels import qe_nc as qe_nc_x, res_nc as res_nc_x

        def gen_qe(n, seed):
            r = qe_nc_x.generate(n, material=cfg.material, seed=seed, return_events=True,
                                 use_achilles_nc_coupling=bool(cfg.achilles_coupl1_quirk),
                                 theta_acc=LACC)["events"]
            return dict(w=np.asarray(r["w"]), k_nu=np.asarray(r["k_nu"]),
                        p_struck=np.asarray(r["p_struck"]), k_lep=np.asarray(r["k_lep"]),
                        p_N=np.asarray(r["p_N"]), p_pi=np.asarray(r["p_pi"]),
                        ppid=np.asarray(r["ppid"], np.int32), ipid=np.asarray(r["ipid"], np.int32),
                        Npid=np.asarray(r["Npid"], np.int32), _raw=r)

        def gen_res(n, seed):
            r = res_nc_x.generate(n, material=cfg.material, seed=seed, return_events=True,
                                  theta_acc=LACC)["events"]
            return dict(w=np.asarray(r["w"]), k_nu=np.asarray(r["k_nu"]),
                        p_struck=np.asarray(r["p_struck"]), k_lep=np.asarray(r["k_lep"]),
                        p_N=np.asarray(r["p_N"]), p_pi=np.asarray(r["p_pi"]),
                        ppid=np.asarray(r["ppid"], np.int32), ipid=np.asarray(r["ipid"], np.int32),
                        Npid=np.asarray(r["Npid"], np.int32), _raw=r)
        m_extra = dict(flux=cfg.flux, theta_acc=list(LACC))
    else:
        from adonis.reweight.reweight_model import build_hv_sf
        from adonis.channels import qe as qe_x, res as res_x
        sf = SpectralFunction(tgt.spectral_n); sf_p = SpectralFunction(tgt.spectral_p)

        def gen_qe(n, seed):
            q = qe_x.sample_importance(n, seed=seed, sf=sf, n_neutron=n_neutron)
            q = _accept_lepton(q, LACC, kkey="k_lep"); nq = len(q["w"])
            return dict(w=np.asarray(q["w"]) / CHUNK, k_nu=np.asarray(q["k_nu"]), p_struck=np.asarray(q["p_struck"]),
                        k_lep=np.asarray(q["k_lep"]), p_N=np.asarray(q["p_out"]), p_pi=np.zeros((nq, 4)),
                        ppid=np.zeros(nq, np.int32), ipid=np.full(nq, 2112, np.int32),
                        Npid=np.full(nq, 2212, np.int32), _raw=q)

        def gen_res(n, seed):
            r = res_x.generate(n, seed=seed, return_events=True, sf_n=sf, sf_p=sf_p,
                               n_neutron=n_neutron, n_proton=n_proton)["events"]
            r = _accept_lepton(r, LACC, kkey="k_lep")
            return dict(w=np.asarray(r["w"]), k_nu=np.asarray(r["k_nu"]), p_struck=np.asarray(r["p_struck"]),
                        k_lep=np.asarray(r["k_lep"]), p_N=np.asarray(r["p_N"]), p_pi=np.asarray(r["p_pi"]),
                        ppid=np.asarray(r["ppid"], np.int32), ipid=np.asarray(r["ipid"], np.int32),
                        Npid=np.asarray(r["Npid"], np.int32), _raw=r)
        m_extra = dict(flux=cfg.flux, theta_acc=list(LACC))

    def cascade(ev, key, caps, chan):
        return CF.cascade_nucleus(jnp.asarray(ev["p_pi"]), jnp.asarray(ev["p_N"]),
                                  jnp.asarray(ev["ppid"]).astype(jnp.int32), jnp.asarray(ev["ipid"]).astype(jnp.int32),
                                  jnp.asarray(ev["Npid"]).astype(jnp.int32),
                                  POOL(seed=(2 if chan == "qe" else 1)), key, channel=chan,
                                  rec_caps=caps, return_fate=True, flat_rec=True)

    def cal_caps(gen, chan):
        """Size the flat FSI record from a calibration run, extrapolated to the chunk.

        The calibration runs the cascade on a small sample and scales linearly to the chunk.  The
        estimate is not conservative: an undersized buffer aborts the generation, and a larger
        nucleus needs a larger ADONIS_REC_MARGIN than the default carries.
        """
        ncal = min(CHUNK, int(os.environ.get("ADONIS_REC_NCAL", "2000")))
        ev = gen(ncal, SEED0); rec = cascade(ev, jax.random.PRNGKey(7), (8, 8), chan)[4]
        scale = CHUNK / ncal * _MARGIN
        t = max(max(64, math.ceil(int(rec["gc_p"]) * scale)), max(64, math.ceil(int(rec["gc_n"]) * scale)))
        return (t, t)

    CAPS_qe = CAPS_res = (64, 64)
    if do_qe: CAPS_qe = cal_caps(gen_qe, "qe"); log(f"flat FSI caps qe -> {CAPS_qe}")
    if do_res: CAPS_res = cal_caps(gen_res, "res"); log(f"flat FSI caps res -> {CAPS_res}")
    manifest = dict(n_chunks=n_chunks, chunk=CHUNK, n_total=CHUNK * n_chunks, material=cfg.material,
                    channels=list(cfg.channels), caps_qe=list(CAPS_qe), caps_res=list(CAPS_res),
                    probe=cfg.probe, achilles_coupl1_quirk=bool(cfg.achilles_coupl1_quirk), **m_extra)

    stage_t = []
    for c in range(n_chunks):
        kq, kr = jax.random.split(jax.random.PRNGKey(1000 + c), 2)
        evs, cols, rec_blocks, out_blocks, ns, pterms = [], [], [], [], [], []
        t_pre = t_cas = 0.0
        for chan, key, caps in ([("qe", kq, CAPS_qe)] if do_qe else []) + ([("res", kr, CAPS_res)] if do_res else []):
            _t0 = time.time()
            ev = (gen_qe if chan == "qe" else gen_res)(CHUNK, SEED0 + c); nb = len(ev["p_N"])
            _t1 = time.time()
            _pt, nt, _o, _cr, rec, pf = cascade(ev, key, caps, chan)
            jax.block_until_ready((nt[0], pf))
            t_pre += _t1 - _t0; t_cas += time.time() - _t1
            evs.append((chan, ev)); cols.append(np.zeros(nb, np.int8) if chan == "qe" else np.ones(nb, np.int8))
            out_blocks.append((nt[0], pf)); rec_blocks.append(CF.compact_fsi_record(dict(rec)))
            ns.append(nb); pterms.append(_pt)
        log(f"chunk {c+1}/{n_chunks}: cascades done ({', '.join('%s=%d' % (e[0], n) for e, n in zip(evs, ns))})"
            f" | pre-FSI {t_pre:.1f}s, cascade {t_cas:.1f}s")
        t_rec0 = time.time()

        save, meta = REC.cascade_outcome_record(out_blocks, rec_blocks, ns)
        if meta["ndrop"]: log(f"chunk {c+1}: dropped {meta['ndrop']} non-physical final-state particles")
        log(f"chunk {c+1}: FSI record -> pion {meta['pion_slots']} slots, nucleon {meta['nucleon_slots']} slots")
        save["channel"] = np.concatenate(cols)

        if EM:
            cat = lambda k: np.concatenate([e[1][k] for e in evs])
            save.update(c=cat("c").astype(np.float64), omega=cat("omega").astype(np.float32),
                        theta=cat("theta").astype(np.float32),
                        k_lep=cat("k_lep").astype(np.float32))
            if do_qe and do_res:
                from adonis.reweight.reweight_model import build_hv_sf
                nq = ns[0]; nr = ns[-1]; qraw = evs[0][1]["_raw"]; rraw = evs[-1][1]["_raw"]
                HV, _SF = build_hv_sf(qraw, rraw, sf, probe="EM")
                save.update(_joint_records(HV, nq, nr, res_identity=True))
                save.update(w0=cat("c").astype(np.float64),
                            p_struck=np.concatenate([np.asarray(qraw["p_struck"]),
                                                     np.asarray(rraw["p_struck"])]).astype(np.float32))
        else:
            nq = ns[0] if do_qe else 0; nr = ns[-1] if do_res else 0
            qref = evs[0][1] if do_qe else None; rref = evs[-1][1] if do_res else None
            save["prim_pi_pid"] = np.concatenate([np.asarray(pt["pid"]) for pt in pterms]).astype(np.int32)
            save.update(w0=np.concatenate([e[1]["w"] for e in evs]),
                        k_nu=np.concatenate([e[1]["k_nu"] for e in evs]).astype(np.float32),
                        p_struck=np.concatenate([e[1]["p_struck"] for e in evs]).astype(np.float32),
                        k_lep=np.concatenate([e[1]["k_lep"] for e in evs]).astype(np.float32))
            if do_qe and do_res and not NC:
                HV, _SF = build_hv_sf(qref["_raw"], rref["_raw"], sf)
                save.update(_joint_records(HV, nq, nr))
                save.update(res_p_N=np.concatenate([np.zeros((nq, 4), np.float32), np.asarray(rref["p_N"], np.float32)]),
                            res_p_pi=np.concatenate([np.zeros((nq, 4), np.float32), np.asarray(rref["p_pi"], np.float32)]),
                            res_ipid=np.concatenate([np.zeros(nq, np.int32), np.asarray(rref["ipid"], np.int32)]),
                            res_ppid=np.concatenate([np.zeros(nq, np.int32), np.asarray(rref["ppid"], np.int32)]))

        np.savez(f"{outdir}/chunk_{c:03d}.npz", **save)
        del evs, out_blocks, save; _gc.collect()
        stage_t.append(dict(chunk=c, n_events=int(sum(ns)), pre_fsi=t_pre, cascade=t_cas,
                            record_io=time.time() - t_rec0))
        log(f"chunk {c+1}/{n_chunks}: written")
    manifest["stage_seconds"] = stage_t
    json.dump(manifest, open(f"{outdir}/manifest.json", "w"), indent=2)
    log(f"DONE: {cfg.probe} bank in {outdir}/ ({n_chunks} chunks)")
    return outdir


def _joint_records(HV, nq, nr, res_identity=False):
    """Bank fields for the reduced-quadratic hard-vertex records.

    A chunk holds the QE events first and the RES events after, so each channel's M is padded with
    zeros over the other's rows: a zero M gives a zero-over-zero guarded ratio of 1, i.e. that channel
    contributes no reweight for events it does not own.
    """
    qe_M = np.zeros((nq + nr, 4, 4), np.float64); qe_Q2 = np.zeros(nq + nr, np.float32)
    res_M = np.zeros((nq + nr, 6, 6), np.float64); res_Q2 = np.zeros(nq + nr, np.float32)
    if nq:
        qe_M[:nq] = np.asarray(HV["qe_reduced"]["M"], np.float64)
        qe_Q2[:nq] = np.asarray(HV["qe_reduced"]["Q2"], np.float32)
    if nr and not res_identity:
        res_M[nq:] = np.asarray(HV["res_reduced"]["M"], np.float64)
        res_Q2[nq:] = np.asarray(HV["res_reduced"]["Q2"], np.float32)
    out = dict(hv_qe_mij=qe_M, hv_qe_Q2=qe_Q2, hv_res_mij=res_M, hv_res_Q2=res_Q2)
    isp = HV.get("qe_isp")
    if isp is not None:
        out["hv_qe_isp"] = np.concatenate([np.asarray(isp, bool), np.zeros(nr, bool)])
        out["hv_qe_probe_em"] = np.int32(1 if HV.get("qe_probe") == "EM" else 0)
    return out


def _generate_hadron(cfg, outdir, log, t0):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.nuclear.targets import resolve_targets
    from adonis.flux.hadron import BEAMS, R_DISK, PIR2_MB, HadronBeam
    from adonis.fsi.cascade import (DiscreteCascadeConfig, _load_density, sample_nucleons, _CH_MASS,
                                    _MP_PHYS, _MN_PHYS)
    import adonis.fsi.cascade as CF

    pid, species, charge = BEAMS[cfg.beam]
    tg = resolve_targets(cfg.material)[0][0]
    ccfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                 step=cfg.cascade.step, pauli=cfg.pauli, nn_inelastic=cfg.cascade.nn_inelastic,
                                 engine="pool")
    _rg, _rp, _rn, radius = _load_density(ccfg.nucleus, ccfg.density_n)
    z0 = -1.05 * float(radius)
    mass = float(_CH_MASS[charge]) if species == "PION" else (_MP_PHYS if charge == 1 else _MN_PHYS)
    hb = HadronBeam(cfg.beam, mass)
    CHUNK = cfg.chunk or cfg.n_per_seed
    n_total = CHUNK * cfg.n_seeds
    n_chunks = cfg.n_seeds
    log(f"beam[{cfg.beam}] pid={pid} {species} q={charge} m={mass:.3f} | |p| in [{cfg.pmin},{cfg.pmax}] "
        f"| N={n_total:,} in {n_chunks} chunk(s) | R_disk={R_DISK} fm (pi R^2 = {PIR2_MB:.1f} mb) -> {outdir}")

    for c in range(n_chunks):
        m = CHUNK
        key = jax.random.PRNGKey(cfg.seed0 + 1000 * c)
        k_mom, k_disk, k_nuc, k_run = jax.random.split(key, 4)
        mom, p4, pos0 = hb.sample(k_mom, k_disk, m, cfg.pmin, cfg.pmax, z0)
        kn, kp = jax.random.split(k_nuc, 2)
        npos, nmom, nisp = sample_nucleons(kn, m, ccfg); A = nisp.shape[1]
        _sc = max(1, -(-A // 12))
        su = dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=jnp.zeros((m, A), bool),
                  ch0=jnp.full(m, charge if species == "PION" else 0, jnp.int32), kp=kp)
        _k1, knuc, _k2 = jax.random.split(su["kp"], 3)
        g0 = CF.empty_batch(m, 1)
        g0["alive"] = jnp.ones((m, 1), bool)
        g0["species"] = jnp.full((m, 1), CF.PION if species == "PION" else CF.NUCLEON, jnp.int32)
        g0["charge"] = jnp.full((m, 1), charge, jnp.int32)
        g0["p4"] = p4[:, None, :]; g0["pos"] = pos0[:, None, :]
        g0["origin"] = jnp.full((m, 1), CF._ORIG_PRIM_PI, jnp.int32)
        g0["external_test"] = jnp.ones((m, 1), bool)
        _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(m))
        g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]
        stepper = CF.make_pool_stepper(su, ccfg, with_rec=True)
        out, _sofl, _oofl, prim_fate, (rec, rofl) = CF.run_cascade_pool(
            g0, stepper, knuc, su["consumed0"], M=12, max_steps=2000, M_out=24,
            prim_origin=CF._ORIG_PRIM_PI, rec_caps=(96 * _sc, 256 * _sc), flat_rec=False)
        assert int(rofl) == 0, f"FSI record overflow in chunk {c}"
        log(f"chunk {c+1}/{n_chunks}: cascade done ({m:,} ev, {time.time()-t0:.0f}s)")

        O = {k: np.asarray(v) for k, v in out.items()}
        comp = CF.compact_fsi_record({k: np.asarray(v) for k, v in rec.items()})
        save, meta = REC.cascade_outcome_record([(O, np.asarray(prim_fate))], [comp], [m])
        save["beam_p"] = np.asarray(mom, np.float32)
        save["w0"] = np.full(m, PIR2_MB, np.float64)
        np.savez(f"{outdir}/chunk_{c:03d}.npz", **save)
        _fl = REC.derive_flags(save)
        log(f"chunk {c+1}/{n_chunks}: written | reacted {_fl['reacted'].mean():.3f} | absorbed "
            f"{_fl['absorbed'].mean():.3f} | <n_pi_out> {save['n_pi_out'].mean():.3f}")
        del O, save, out
    json.dump(dict(beam=cfg.beam, pid=pid, species=species, charge=charge, pmin=cfg.pmin, pmax=cfg.pmax,
                   n_total=n_total, n_chunks=n_chunks, R_disk=R_DISK, pir2_mb=PIR2_MB, material=cfg.material,
                   seed0=cfg.seed0, probe=cfg.probe), open(f"{outdir}/manifest.json", "w"), indent=1)
    log(f"DONE: hadron bank in {outdir}/ ({n_chunks} chunks)")
    return outdir


def load_bank(outdir, max_chunks=None):
    """Concatenate the chunk_*.npz of a bank dir into one in-memory dict (+ ["manifest"]).  Ragged
    per-slot event indices (FSI f_*_eidx, escaped-list ks_eidx) are offset into the global event
    numbering; everything else -- including prim_fate (n, Wmax) and nsc_prim (n,) -- concatenates on
    axis 0."""
    man = json.load(open(f"{outdir}/manifest.json"))
    files = sorted(glob.glob(f"{outdir}/chunk_*.npz"))
    if max_chunks is not None:
        files = files[:max_chunks]
    per, ev_off = {}, 0
    for f in files:
        d = np.load(f)
        nev = len(d["prim_fate"])
        for k in d.files:
            v = d[k]
            if k in ("f_p_eidx", "f_n_eidx", "ks_eidx"):
                v = v.astype(np.int64) + ev_off
            per.setdefault(k, []).append(v)
        ev_off += nev
    B = {k: np.concatenate(v) for k, v in per.items()}
    B["manifest"] = man
    return B
