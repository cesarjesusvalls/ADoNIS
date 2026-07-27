"""generate_reweight_bank(GenConfig): THE config-driven reweight-bank builder (probe weak | EM).

ONE driver, dispatched by `adonis.workflow.cli`, for the differentiable EVENT BANKS that store the FULL
per-event REWEIGHT RECORDS (so the exact w(theta) is available for every knob at any theta -- no Taylor).
Unifies the two legacy env-driven generators that this replaces (both retired):
  * weak (CC neutrino): was `adonis/reweight/event_bank.py` -- qe/res primaries, HARD-VERTEX amps2
    records (hv_*), FSI kind-1 record (f_*), full post-FSI final state (fs_*).
  * EM   (electron)   : was `analysis/beams/ee_event_bank.py` -- ee/res_ee primaries, (e,e') weight c
    + leptonic omega/theta, FSI kind-1 record (f_*), fs_*.  No amps2 (EM form-factor knobs are
    buildable later from the stored primary kinematics).

Byte-for-byte identical to both legacy generators for the same (material, seed0, n_per_seed, n_seeds):
this is a structural refactor only -- same samplers, same per-chunk RNG (cascade keys
split(PRNGKey(1000+c)); pool seeds qe=2/res=1), same records.  The FSI-record CAP calibration (buffer
size) may differ harmlessly -- `compact_fsi_record` trims to the actual slot count, so the stored bytes
do not depend on the cap as long as it does not overflow.

Layout (matches paper_banks/): channel 0=qe,1=res concatenated; chunk_NNN.npz + manifest.json.  One
seed per chunk (seed = seed0 + c), so SLURM shards over seed0 stay independent, exactly as before.
"""
from __future__ import annotations
import os, time, json, gc as _gc, math
from pathlib import Path
import numpy as np

PMAX = 1e4   # MeV ceiling: drop the rare (~0.1%) cascade-artifact nucleons (inf/sentinel momenta)


def _idma(n):  # 4-component identity (a,b,c,Q2)=(1,0,0,1): ma_reweight & strength_reweight both -> 1
    return [np.ones(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), np.ones(n, np.float32)]


def _compact_fs(nt):
    """Post-FSI escaped-finals buffer -> ragged (off, pid, chg, p4) + count of dropped non-physical rows.
    IDENTICAL in both legacy generators (verbatim)."""
    al = np.asarray(nt["alive"]); chg = np.asarray(nt["charge"]); spc = np.asarray(nt["species"])
    p4 = np.asarray(nt["p4"]).astype(np.float64)
    pid = np.where(spc == 0, np.array([211, 111, -211])[np.clip(chg, 0, 2)], np.where(chg == 1, 2212, 2112))
    mom = np.sqrt(np.nan_to_num(p4[:, :, 1:] ** 2, posinf=np.inf).sum(2))
    good = al & np.isfinite(p4).all(2) & (mom < PMAX)
    cnt = good.sum(1).astype(np.int64); off = np.concatenate([[0], np.cumsum(cnt)]); m = good.reshape(-1)
    return off, pid.reshape(-1)[m].astype(np.int32), chg.reshape(-1)[m].astype(np.int32), \
        p4.reshape(-1, 4)[m].astype(np.float32), int((al & ~good).sum())


def _merge_off(a, b):
    return np.concatenate([a, a[-1] + b[1:]]).astype(np.int64)


def _flat_fsi_save(recs, ns):
    """Concatenate the per-block FLAT FSI kind-1 records (already compacted), offsetting each block's
    per-slot event index by the running event count, and cast to the stored dtypes.  IDENTICAL logic in
    both legacy generators (QE block is events 0..nq-1, RES block nq..nq+nr-1)."""
    from adonis.fsi.cascade import _P_SLOT, _N_SLOT
    flat = {f: np.concatenate([rc[f] for rc in recs]) for f in _P_SLOT + _N_SLOT}
    for tag in ("p_eidx", "n_eidx"):
        acc, base = [], 0
        for rc, nn in zip(recs, ns):
            acc.append(np.asarray(rc[tag]) + base); base += nn
        flat[tag] = np.concatenate(acc)
    fsi_save = {"f_p_eidx": flat["p_eidx"].astype(np.int32), "f_n_eidx": flat["n_eidx"].astype(np.int32)}
    for field, arr in flat.items():
        if field.endswith("_eidx"):
            continue
        dt = (np.int8 if field in ("bc", "iso")
              else (bool if field in ("hh", "inel", "swap", "pi_hh") else np.float32))
        fsi_save[f"f_{field}"] = arr.astype(dt)
    return fsi_save, len(flat["p_eidx"]), len(flat["n_eidx"])


def generate_reweight_bank(gc, outdir, log=None):
    """Build the reweight bank described by GenConfig `gc` into `outdir` (chunk_NNN.npz + manifest.json).

    gc.n_per_seed = events per chunk (dense FSI buffers scale with THIS); gc.n_seeds = number of chunks;
    seed = gc.seed0 + c per chunk.  gc.probe selects the leptonic current (weak: qe/res + amps2; EM:
    ee/res_ee).  gc.channels selects which of {qe,res} to bank (both for the canonical paper banks).
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.reweight import tune as T
    from adonis.workflow.materials import resolve_targets
    from adonis.nuclear.spectral import SpectralFunction
    import adonis.fsi.cascade as CF

    EM = (gc.probe == "EM")
    CHUNK = gc.n_per_seed
    n_chunks = gc.n_seeds
    SEED0 = gc.seed0
    MATERIAL = gc.material
    do_qe = "qe" in gc.channels
    do_res = "res" in gc.channels
    t0 = time.time()
    if log is None:
        def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    # weak/spectrum: the neutrino flux is the single source from gc.flux -> export ADONIS_FLUX_FILE so the
    # generators' SpectrumFlux() reads the matching table (GenConfig.__post_init__ already asserts consistency).
    if not EM:
        from adonis.workflow.config import FLUX_FILES
        os.environ["ADONIS_FLUX_FILE"] = FLUX_FILES[gc.flux]

    tgt = resolve_targets(MATERIAL)[0][0]
    n_neutron = tgt.A - tgt.Z; n_proton = tgt.Z
    CF.FLAT_FSI_REC = True                                  # both banks always FLAT/streaming record
    _MARGIN = float(os.environ.get("ADONIS_REC_MARGIN", "1.5"))
    POOL = lambda **k: T.POOLCFG(nucleus=tgt.density_p, density_n=tgt.density_n, configs=tgt.configs, **k)

    os.makedirs(outdir, exist_ok=True)

    # ---- probe-specific primary samplers -> a COMMON event dict ready for the cascade ----
    # Every gen returns keys: cascade inputs (p_pi, p_N, ppid, ipid, Npid) + whatever the save needs.
    if EM:
        from adonis.channels import ee as ee_xsec, res_ee as res_ee_xsec
        E_BEAM = float(gc.e_beam)

        def gen_qe(n, seed):
            r = ee_xsec.generate(n, material=MATERIAL, seed=seed, E_beam=E_BEAM, records=True, theta_acc=(0.0, 180.0))
            pid = np.where(r["is_p"], 2212, 2112).astype(np.int32)   # EM: struck species unchanged
            return dict(c=np.asarray(r["c"]), omega=np.asarray(r["omega"]), theta=np.asarray(r["theta"]),
                        p_N=np.asarray(r["p_out"]), p_pi=np.zeros((len(pid), 4)),
                        ppid=np.zeros(len(pid), np.int32), ipid=pid, Npid=pid)

        def gen_res(n, seed):
            r = res_ee_xsec.generate(n, material=MATERIAL, seed=seed, E_beam=E_BEAM, records=True)
            return dict(c=np.asarray(r["c"]), omega=np.asarray(r["omega"]), theta=np.asarray(r["theta"]),
                        p_N=np.asarray(r["p_N"]), p_pi=np.asarray(r["p_pi"]),
                        ppid=np.asarray(r["ppid"], np.int32), ipid=np.asarray(r["ipid"], np.int32),
                        Npid=np.asarray(r["Npid"], np.int32))
        manifest_extra = dict(probe="ee", E_beam=E_BEAM)
    else:
        from adonis.reweight.reweight_model import build_hv_sf
        from adonis.channels import qe as qe_xsec, res as res_xsec
        sf = SpectralFunction(tgt.spectral_n); sf_p = SpectralFunction(tgt.spectral_p)

        def gen_qe(n, seed):
            q = qe_xsec.sample_importance(n, seed=seed, sf=sf, n_neutron=n_neutron)
            nq = len(q["w"])
            return dict(w=np.asarray(q["w"]) / CHUNK, k_nu=np.asarray(q["k_nu"]),
                        p_struck=np.asarray(q["p_struck"]), k_mu=np.asarray(q["k_mu"]),
                        p_N=np.asarray(q["p_out"]), p_pi=np.zeros((nq, 4)),
                        ppid=np.zeros(nq, np.int32), ipid=np.full(nq, 2112, np.int32),
                        Npid=np.full(nq, 2212, np.int32), _raw=q)

        def gen_res(n, seed):
            r = res_xsec.generate(n, seed=seed, return_events=True, sf_n=sf, sf_p=sf_p,
                                  n_neutron=n_neutron, n_proton=n_proton)["events"]
            return dict(w=np.asarray(r["w"]), k_nu=np.asarray(r["k_nu"]), p_struck=np.asarray(r["p_struck"]),
                        k_mu=np.asarray(r["k_mu"]), p_N=np.asarray(r["p_N"]), p_pi=np.asarray(r["p_pi"]),
                        ppid=np.asarray(r["ppid"], np.int32), ipid=np.asarray(r["ipid"], np.int32),
                        Npid=np.asarray(r["Npid"], np.int32), _raw=r)
        manifest_extra = dict(probe="weak", flux=gc.flux)

    def cascade(ev, key, caps, chan):
        return CF.cascade_nucleus(jnp.asarray(ev["p_pi"]), jnp.asarray(ev["p_N"]),
                                  jnp.asarray(ev["ppid"]).astype(jnp.int32), jnp.asarray(ev["ipid"]).astype(jnp.int32),
                                  jnp.asarray(ev["Npid"]).astype(jnp.int32),
                                  POOL(seed=(2 if chan == "qe" else 1)), key, channel=chan, rec_caps=caps)

    # ---- FSI-record cap calibration: MEASURE this config's per-event interaction rate, x margin. ----
    # The buffer size does NOT affect the stored bytes (compact trims to actual slots); it only must not
    # overflow.  Measured on a small sample with a throwaway (8,8) counter buffer.
    def cal_caps(gen, chan):
        ncal = min(CHUNK, int(os.environ.get("ADONIS_REC_NCAL", "100")))
        ev = gen(ncal, SEED0)
        rec = cascade(ev, jax.random.PRNGKey(7), (8, 8), chan)[4]
        scale = CHUNK / ncal * _MARGIN

        def b(g):
            return max(64, math.ceil(int(g) * scale))
        t = max(b(rec["gc_p"]), b(rec["gc_n"]))            # symmetric max (pion count zero-inflated; see legacy)
        return (t, t)

    CAPS_qe = CAPS_res = (64, 64)
    if do_qe:
        CAPS_qe = cal_caps(gen_qe, "qe"); log(f"flat FSI caps qe -> {CAPS_qe}")
    if do_res:
        CAPS_res = cal_caps(gen_res, "res"); log(f"flat FSI caps res -> {CAPS_res}")

    manifest = dict(n_chunks=n_chunks, chunk=CHUNK, n_total=CHUNK * n_chunks, material=MATERIAL,
                    channels=list(gc.channels), caps_qe=list(CAPS_qe), caps_res=list(CAPS_res), **manifest_extra)

    for c in range(n_chunks):
        kq, kr = jax.random.split(jax.random.PRNGKey(1000 + c), 2)
        blocks = []              # (chan, ev, channel_col, (off,pid,chg,p4,ndrop), compacted_fsi_rec)
        if do_qe:
            q = gen_qe(CHUNK, SEED0 + c); nq = len(q["p_N"])
            _pt, ntq, _o, _cq, recq = cascade(q, kq, CAPS_qe, "qe")
            blocks.append(("qe", q, np.zeros(nq, np.int8), _compact_fs(ntq[0]),
                           CF.compact_fsi_record(dict(recq)), _pt))
        if do_res:
            r = gen_res(CHUNK, SEED0 + c); nr = len(r["p_N"])
            _pt, ntr, _o2, _cr, recr = cascade(r, kr, CAPS_res, "res")
            blocks.append(("res", r, np.ones(nr, np.int8), (_compact_fs(ntr[0]) if True else None),
                           CF.compact_fsi_record(dict(recr)), _pt))
        _sizes = ", ".join("%s=%d" % (b[0], len(b[1]["p_N"])) for b in blocks)
        log(f"chunk {c+1}/{n_chunks}: cascades done ({_sizes})")

        # concat final state + FSI record across present blocks (QE first, then RES)
        offs = [b[3][0] for b in blocks]; off = offs[0]
        for o in offs[1:]:
            off = _merge_off(off, o)
        fs_pid = np.concatenate([b[3][1] for b in blocks]); fs_chg = np.concatenate([b[3][2] for b in blocks])
        fs_p4 = np.concatenate([b[3][3] for b in blocks]); ndrop = sum(b[3][4] for b in blocks)
        if ndrop:
            log(f"chunk {c+1}: dropped {ndrop} non-physical final-state particles")
        ns = [len(b[1]["p_N"]) for b in blocks]
        fsi_save, npslot, nnslot = _flat_fsi_save([b[4] for b in blocks], ns)
        log(f"chunk {c+1}: FSI record streamed -> pion {npslot} slots, nucleon {nnslot} slots")

        channel = np.concatenate([b[2] for b in blocks])
        save = dict(channel=channel, fs_off=off, fs_pid=fs_pid, fs_chg=fs_chg, fs_p4=fs_p4, **fsi_save)

        if EM:
            cat = lambda key: np.concatenate([b[1][key] for b in blocks])
            save.update(c=cat("c").astype(np.float64), omega=cat("omega").astype(np.float32),
                        theta=cat("theta").astype(np.float32))
        else:
            # weak: w0 + primary kinematics + RES vertex + prim-pion fate + HARD-VERTEX amps2 records.
            nq = ns[0] if do_qe else 0
            nr = ns[-1] if do_res else 0
            qref = blocks[0][1] if do_qe else None
            rref = blocks[-1][1] if do_res else None
            prim_pi = np.concatenate([np.asarray(b[5]["pid"]) for b in blocks]).astype(np.int32)
            save.update(prim_pi_pid=prim_pi,
                        w0=np.concatenate([b[1]["w"] for b in blocks]),
                        k_nu=np.concatenate([b[1]["k_nu"] for b in blocks]).astype(np.float32),
                        p_struck=np.concatenate([b[1]["p_struck"] for b in blocks]).astype(np.float32),
                        k_mu=np.concatenate([b[1]["k_mu"] for b in blocks]).astype(np.float32))
            if do_qe and do_res:
                HV, _SF = build_hv_sf(qref["_raw"], rref["_raw"], sf, with_pw=False)

                def hv_qeonly(rec):
                    return [np.concatenate([np.asarray(rec[i], np.float32), _idma(nr)[i]]) for i in range(4)]

                def hv_resonly(rec):
                    return [np.concatenate([_idma(nq)[i], np.asarray(rec[i], np.float32)]) for i in range(4)]
                qe_ma = [np.concatenate([np.asarray(HV["qe_ma"][i], np.float32), _idma(nr)[i]]) for i in range(4)]
                res_ma = [np.concatenate([_idma(nq)[i], np.asarray(HV["res_ma"][i], np.float32)]) for i in range(4)]
                hv = dict(qe_ma=qe_ma, res_ma=res_ma, qe_vec=hv_qeonly(HV["qe_vec"]),
                          qe_gmp=hv_qeonly(HV["qe_gmp"]), qe_gmn=hv_qeonly(HV["qe_gmn"]),
                          qe_gep=hv_qeonly(HV["qe_gep"]), qe_gen=hv_qeonly(HV["qe_gen"]),
                          res_pp=hv_resonly(HV["res_pp"]), res_delta=hv_resonly(HV["res_delta"]))
                save.update({f"hv_{name}_{abc}": comp[i].astype(np.float32)
                             for name, comp in hv.items()
                             for i, abc in enumerate(["a", "b", "c", "Q2"][:len(comp)])})
                save.update(res_p_N=np.concatenate([np.zeros((nq, 4), np.float32), np.asarray(rref["p_N"], np.float32)]),
                            res_p_pi=np.concatenate([np.zeros((nq, 4), np.float32), np.asarray(rref["p_pi"], np.float32)]),
                            res_ipid=np.concatenate([np.zeros(nq, np.int32), np.asarray(rref["ipid"], np.int32)]),
                            res_ppid=np.concatenate([np.zeros(nq, np.int32), np.asarray(rref["ppid"], np.int32)]))

        np.savez(f"{outdir}/chunk_{c:03d}.npz", **save)
        del blocks, save; _gc.collect()
        log(f"chunk {c+1}/{n_chunks}: written ({len(channel)} events)")

    with open(f"{outdir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE: reweight bank ({manifest_extra['probe']}) in {outdir}/ ({n_chunks} chunks) + manifest.json")
    return outdir
