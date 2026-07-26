"""(e,e') differentiable EVENT BANK with full FSI reweight records (EM probe).

Sibling of analysis/t2k/differentiability/event_bank.py, for the ELECTRON probe.  QE via ee_xsec, RES via
res_ee_xsec.  Per event stores: the (e,e') contribution c [nb] (SUM = sigma_inclusive), the leptonic
observables omega/theta_e', the channel (0=qe,1=res), the POST-FSI hadron list (ragged fs_*), and the FLAT
FSI reweight record (f_*, p_eidx/n_eidx) -- so FSI knobs reweight ANY (semi-)exclusive signal downstream,
exactly as for the neutrino bank.  No HV/form-factor records (FSI-focused; EM-FF knobs can be added later
from the stored primary kinematics).  All angles are kept (theta_acc=(0,180)); slice downstream.

  EE_N=100000 CHUNK=100000 SEED0=0 ADONIS_MATERIAL=C python -m analysis.beams.ee_event_bank <outdir>
"""
import os, sys, time, json, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

PMAX = 1e4   # MeV ceiling: drop rare cascade-artifact nucleons (mirror event_bank)


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import math
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    outdir = argv[0] if argv else "output/ee_event_bank"
    sys.argv = [sys.argv[0], "dpt"]                       # tune.py import guard (reads argv[1])
    from analysis.t2k.differentiability import tune as T
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec import ee_xsec, res_ee_xsec
    import adonis.fsi.cascade as CF
    from adonis.fsi.cascade import _P_SLOT, _N_SLOT

    N_TOTAL = int(os.environ.get("EE_N", "100000"))       # samples per species/channel per chunk
    CHUNK = min(int(os.environ.get("CHUNK", str(N_TOTAL))), N_TOTAL)
    n_chunks = int(np.ceil(N_TOTAL / CHUNK))
    SEED0 = int(os.environ.get("SEED0", "0"))
    MATERIAL = os.environ.get("ADONIS_MATERIAL", "C")
    CH = os.environ.get("EE_CH", "both")                  # qe | res | both
    E_BEAM = float(os.environ.get("EE_EBEAM", str(ee_xsec.E_BEAM_JLAB)))
    tgt = resolve_targets(MATERIAL)[0][0]
    POOL = lambda **k: T.POOLCFG(nucleus=tgt.density_p, density_n=tgt.density_n, configs=tgt.configs, **k)
    CF.FLAT_FSI_REC = True                                 # always flat/streaming record here
    _MARGIN = float(os.environ.get("ADONIS_REC_MARGIN", "1.5"))

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    log(f"ee_event_bank ({CH}): N={N_TOTAL}/species x {n_chunks} chunk(s), {MATERIAL}, E={E_BEAM} MeV -> {outdir}")

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

    def cascade(ev, key, caps, chan):
        n = len(ev["c"])
        return CF.cascade_nucleus(jnp.asarray(ev["p_pi"]), jnp.asarray(ev["p_N"]),
                                  jnp.asarray(ev["ppid"]), jnp.asarray(ev["ipid"]), jnp.asarray(ev["Npid"]),
                                  POOL(seed=(2 if chan == "qe" else 1)), key, channel=chan, rec_caps=caps)

    def cal_caps(gen, chan):
        ncal = min(CHUNK, int(os.environ.get("ADONIS_REC_NCAL", "100")))
        ev = gen(ncal, SEED0)
        rec = cascade(ev, jax.random.PRNGKey(7), (8, 8), chan)[4]   # throwaway counter buffer
        # Total slots scale with the per-species INPUT count (gen(x) emits ~mult*x events, mult=#species/
        # channels); so scale the calibration count by CHUNK/ncal directly (species-multiplicity-robust) --
        # NOT by CHUNK/len(ev) which would undershoot by the multiplicity.
        scale = CHUNK / ncal * _MARGIN
        def b(g): return max(64, math.ceil(int(g) * scale))
        t = max(b(rec["gc_p"]), b(rec["gc_n"]))                     # symmetric max (see event_bank rationale)
        n = max(len(ev["c"]), 1)
        return (t, t), (int(rec["gc_p"]) / n, int(rec["gc_n"]) / n)

    do_qe, do_res = CH in ("qe", "both"), CH in ("res", "both")
    CAPS_qe = CAPS_res = (64, 64)
    if do_qe:
        CAPS_qe, rq = cal_caps(gen_qe, "qe"); log(f"cal qe rate=({rq[0]:.2f},{rq[1]:.2f})/ev -> {CAPS_qe}")
    if do_res:
        CAPS_res, rr = cal_caps(gen_res, "res"); log(f"cal res rate=({rr[0]:.2f},{rr[1]:.2f})/ev -> {CAPS_res}")

    def compact_fs(nt):
        al = np.asarray(nt["alive"]); chg = np.asarray(nt["charge"]); spc = np.asarray(nt["species"])
        p4 = np.asarray(nt["p4"]).astype(np.float64)
        pid = np.where(spc == 0, np.array([211, 111, -211])[np.clip(chg, 0, 2)], np.where(chg == 1, 2212, 2112))
        mom = np.sqrt(np.nan_to_num(p4[:, :, 1:] ** 2, posinf=np.inf).sum(2))
        good = al & np.isfinite(p4).all(2) & (mom < PMAX)
        cnt = good.sum(1).astype(np.int64); off = np.concatenate([[0], np.cumsum(cnt)]); m = good.reshape(-1)
        return off, pid.reshape(-1)[m].astype(np.int32), chg.reshape(-1)[m].astype(np.int32), \
            p4.reshape(-1, 4)[m].astype(np.float32), int((al & ~good).sum())

    def merge_off(a, b): return np.concatenate([a, a[-1] + b[1:]]).astype(np.int64)

    manifest = dict(probe="ee", material=MATERIAL, E_beam=E_BEAM, channels=CH, n_chunks=n_chunks,
                    chunk=CHUNK, caps_qe=list(CAPS_qe), caps_res=list(CAPS_res))
    for c in range(n_chunks):
        kq, kr = jax.random.split(jax.random.PRNGKey(1000 + c), 2)
        blocks, fs_parts, fsi_parts, nqs = [], [], {"p": [], "n": []}, []
        eoff = 0     # running event-index offset for the flat FSI eidx across QE then RES
        cq = cr = None
        # ---- QE block ----
        if do_qe:
            q = gen_qe(CHUNK, SEED0 + c); nq = len(q["c"])
            ptq, ntq, _o, _cq, recq = cascade(q, kq, CAPS_qe, "qe")
            offq, fpq, fcq, fp4q, dq = compact_fs(ntq[0])
            cqrec = CF.compact_fsi_record(dict(recq))
            blocks.append(("qe", q, np.zeros(nq, np.int8), (offq, fpq, fcq, fp4q, dq)))
            cq = cqrec; nqs.append(nq)
        # ---- RES block ----
        if do_res:
            r = gen_res(CHUNK, SEED0 + c); nr = len(r["c"])
            ptr, ntr, _o2, _cr, recr = cascade(r, kr, CAPS_res, "res")
            offr, fpr, fcr, fp4r, dr = compact_fs(ntr[0])
            crrec = CF.compact_fsi_record(dict(recr))
            blocks.append(("res", r, np.ones(nr, np.int8), (offr, fpr, fcr, fp4r, dr)))
            cr = crrec; nqs.append(nr)
        _sizes = ", ".join("%s=%d" % (b[0], len(b[1]["c"])) for b in blocks)
        log(f"chunk {c+1}/{n_chunks}: cascades done ({_sizes})")

        # concatenate observables + weights + final state across the present blocks (QE first, then RES)
        cat = lambda key: np.concatenate([b[1][key] for b in blocks])
        channel = np.concatenate([b[2] for b in blocks])
        offs = [b[3][0] for b in blocks]; off = offs[0]
        for o in offs[1:]:
            off = merge_off(off, o)
        fs_pid = np.concatenate([b[3][1] for b in blocks]); fs_chg = np.concatenate([b[3][2] for b in blocks])
        fs_p4 = np.concatenate([b[3][3] for b in blocks]); ndrop = sum(b[3][4] for b in blocks)
        if ndrop: log(f"chunk {c+1}: dropped {ndrop} non-physical final-state particles")

        # FLAT FSI record: concat present blocks, offsetting RES eidx by the QE event count
        recs = [cq, cr] if (do_qe and do_res) else ([cq] if do_qe else [cr])
        ns = nqs
        flat = {f: np.concatenate([rc[f] for rc in recs]) for f in _P_SLOT + _N_SLOT}
        for tag in ("p_eidx", "n_eidx"):
            acc, base = [], 0
            for rc, nn in zip(recs, ns):
                acc.append(np.asarray(rc[tag]) + base); base += nn
            flat[tag] = np.concatenate(acc)
        fsi_save = {"f_p_eidx": flat["p_eidx"].astype(np.int32), "f_n_eidx": flat["n_eidx"].astype(np.int32)}
        for field, arr in flat.items():
            if field.endswith("_eidx"): continue
            dt = (np.int8 if field in ("bc", "iso")
                  else (bool if field in ("hh", "inel", "swap", "pi_hh") else np.float32))
            fsi_save[f"f_{field}"] = arr.astype(dt)
        log(f"chunk {c+1}: FSI record streamed -> pion {len(flat['p_eidx'])} slots, "
            f"nucleon {len(flat['n_eidx'])} slots")

        np.savez(f"{outdir}/chunk_{c:03d}.npz",
                 channel=channel, c=cat("c").astype(np.float64),
                 omega=cat("omega").astype(np.float32), theta=cat("theta").astype(np.float32),
                 fs_off=off, fs_pid=fs_pid, fs_chg=fs_chg, fs_p4=fs_p4, **fsi_save)
        del blocks, flat; gc.collect()
        log(f"chunk {c+1}/{n_chunks}: written ({len(channel)} events)")
    with open(f"{outdir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE: ee bank in {outdir}/ ({n_chunks} chunks) + manifest.json")


if __name__ == "__main__":
    run()
