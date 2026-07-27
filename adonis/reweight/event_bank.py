"""Build a persistent DIFFERENTIABLE EVENT BANK that stores the FULL per-event REWEIGHT RECORDS, so the
EXACT reweight w(theta) is available for every knob at any theta -- no Taylor anywhere.  Everything (forward
distributions, gradients via autodiff, exact ratios) is computed at plot time from these records via
bank_reweight.bank_weight (which reproduces model_hist_full's per-event weight exactly).

Per event (QE + RES concatenated; channel 0=qe,1=res), chunked/streamed:
  * kinematics    : k_nu, p_struck, k_mu                                  [f32]
  * final state   : full post-FSI hadron list (pid,charge,p4), RAGGED    [f32]  -> any signal def
  * prim_pi_pid   : primary-pion fate (0=absorbed/none, +-211/111, -1)   [i32]
  * bare weight   : w0 (no signal folded in)                             [f64]
  * HARD-VERTEX amps2 records (a,b,c[,Q2]) -- identity (1,0,0[,1]) on the OTHER channel so the per-event
    reweight is channel-correct with no masking: qe_ma,res_ma (M_A both; axial/res_axial per channel),
    qe_vec, qe_gmp,qe_gmn,qe_gep,qe_gen (QE), res_pp (RES).                [f32]
  * FSI kind-1 record (pion Kp=32: bc,sa,ss_el,ss,si,nh; nucleon Kn=64: hh,a,iso,finel,inel,swap,ns) [mixed]

  CC0PI_N=100000 CHUNK=100000 python analysis/t2k/differentiability/event_bank.py [outdir]
"""
import os, sys, time, json, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

OUTDIR = "output/event_bank"
PMAX = 1e4   # MeV ceiling: drop the rare (~0.1%) cascade artifact nucleons (inf/sentinel momenta)


def _idma(n):  # 4-component identity (a,b,c,Q2)=(1,0,0,1): ma_reweight & strength_reweight both -> 1
    return [np.ones(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), np.ones(n, np.float32)]


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.argv_kept = [a for a in sys.argv[1:] if not a.startswith("-")]
    outdir = sys.argv_kept[0] if sys.argv_kept else OUTDIR
    sys.argv = [sys.argv[0], "dpt"]
    from adonis.reweight import tune as T
    from adonis.reweight.reweight_model import nominal_knobs, build_hv_sf
    from adonis.workflow.materials import resolve_targets
    from adonis.nuclear.spectral import SpectralFunction
    from adonis.channels import qe as qe_xsec, res as res_xsec
    import adonis.fsi.cascade as CF

    N_TOTAL = int(os.environ.get("CC0PI_N", "100000"))
    CHUNK = min(int(os.environ.get("CHUNK", str(N_TOTAL))), N_TOTAL)
    n_chunks = int(np.ceil(N_TOTAL / CHUNK))
    SEED0 = int(os.environ.get("SEED0", "0"))   # per-shard seed offset: each SLURM array task owns a
    #   disjoint seed range (seed = SEED0 + c) so shards produce INDEPENDENT events (default 0 = original).
    # flux + material are env-configurable (T2K/C default -> byte-identical).  ADONIS_FLUX_FILE is read
    # by the generators' T2KFlux() (DEFAULT_FLUX); ADONIS_MATERIAL selects the nucleus (C | Ar).
    MATERIAL = os.environ.get("ADONIS_MATERIAL", "C")
    tgt = resolve_targets(MATERIAL)[0][0]
    sf = SpectralFunction(tgt.spectral_n); sf_p = SpectralFunction(tgt.spectral_p)
    n_neutron = tgt.A - tgt.Z; n_proton = tgt.Z
    NOM = nominal_knobs()
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    log(f"event_bank (full records): N={N_TOTAL} in {n_chunks} chunk(s) of {CHUNK} -> {outdir}")
    log(f"  flux={os.environ.get('ADONIS_FLUX_FILE','flux/T2K_nu.dat')}  material={tgt.symbol}{tgt.A} "
        f"(Z={tgt.Z} N={n_neutron})")
    # TODO(tech-debt, COMPUTATIONAL not physics): the nucleus-scaled fixed cap over-allocates the
    # transient dense buffer for the long occupancy tail (~97% padding).  Proper fix = auto-size K from
    # a pre-pass, or spill rare overflows.  See docs/logbook/fsi_record_cap_techdebt.md.
    # In-slab FSI-record caps must SCALE WITH THE NUCLEUS (heavier target -> longer cascades -> more
    # in-slab steps per event).  Mirror the PROVEN beam_bank.py scaling (_sc=ceil(A/12), base (96,256)):
    # 12C -> (96,256), 40Ar -> (384,1024).  Measured event-bank Ar max occupancy is (pion 92, nucleon 333)
    # -- well inside (384,1024).  Only the TRANSIENT dense buffer grows with the cap; the stored ragged
    # record does not.  The carbon-tuned (96,64) and my earlier flat (256,256) both overflowed on Ar.
    # compact_fsi_record now fails LOUD with the exact K needed if this is ever still too small.
    _sc = max(1, -(-tgt.A // 12))                    # ceil(A/12)  (dense fallback caps only)
    # FLAT (streaming) record: rec_caps is the TOTAL interaction budget = n * per-event-mean * margin,
    # NOT a per-event K (the tail-free fix, docs/logbook/fsi_record_cap_techdebt.md).  We do NOT GUESS the
    # mean -- a small calibration pass MEASURES it for THIS exact config (below, after POOL), x _MARGIN.
    # The loud guard in compact_fsi_record still backstops any pathological tail.
    _flat = os.environ.get("ADONIS_FLAT_FSI", "1") != "0"   # event_bank defaults to FLAT (validated
    CF.FLAT_FSI_REC = _flat                                 #   bit-identical to dense).  ADONIS_FLAT_FSI=0 -> dense.
    _MARGIN = float(os.environ.get("ADONIS_REC_MARGIN", "1.5"))
    CAPS_qe = CAPS_res = (96 * _sc, 256 * _sc)             # legacy per-event dense caps (used when not _flat)
    if os.environ.get("ADONIS_REC_CAPS"):                 # explicit override -> both channels, skip cal
        CAPS_qe = CAPS_res = tuple(int(x) for x in os.environ["ADONIS_REC_CAPS"].split(","))
    # material-aware cascade: thread the nucleus density/configs into the pool config (carbon default).
    POOL = lambda **k: T.POOLCFG(nucleus=tgt.density_p, density_n=tgt.density_n, configs=tgt.configs, **k)
    FSI_F = ("bc", "sa", "ss_el", "ss", "si", "pi_hh", "pi_a", "sa_c", "ss_el_c", "ss_c", "si_c", "nh",
             "hh", "a", "iso", "finel", "inel", "swap", "ns")

    if _flat and not os.environ.get("ADONIS_REC_CAPS"):
        # Auto-size the flat TOTAL budget: MEASURE this config's per-event interaction rate on a small
        # sample, x _MARGIN.  The calibration buffer size is IRRELEVANT -- the flat cursor gc counts every
        # interaction truthfully even when the writes overflow and drop -- so we run it with a throwaway
        # (8,8) buffer purely as a COUNTER.  No chicken-and-egg: we never need a cap to measure the cap.
        import math
        ncal = min(CHUNK, int(os.environ.get("ADONIS_REC_NCAL", "100")))
        qc = qe_xsec.sample_importance(ncal, seed=SEED0, sf=sf, n_neutron=n_neutron)
        rc = res_xsec.generate(ncal, seed=SEED0, return_events=True, sf_n=sf, sf_p=sf_p,
                               n_neutron=n_neutron, n_proton=n_proton)["events"]
        kcq, kcr = jax.random.split(jax.random.PRNGKey(7), 2)
        nqc, nrc = len(qc["w"]), len(rc["w"])
        rq = CF.cascade_nucleus(jnp.zeros((nqc, 4)), jnp.asarray(qc["p_out"]), jnp.zeros(nqc, jnp.int32),
                                jnp.full(nqc, 2112, jnp.int32), jnp.full(nqc, 2212, jnp.int32),
                                POOL(seed=2), kcq, channel="qe", rec_caps=(8, 8))[4]
        rr = CF.cascade_nucleus(jnp.asarray(rc["p_pi"]), jnp.asarray(rc["p_N"]),
                                jnp.asarray(rc["ppid"]).astype(jnp.int32), jnp.asarray(rc["ipid"]).astype(jnp.int32),
                                jnp.asarray(rc["Npid"]).astype(jnp.int32), POOL(seed=1), kcr, channel="res", rec_caps=(8, 8))[4]
        # SYMMETRIC cap: size BOTH buffers to max(pion, nucleon) of the per-cal point estimate * margin.  The
        # pion slot-count is zero-INFLATED and UNMEASURABLE at 100 events -- pions are made by rare threshold
        # NN->NNpi events (cascade.py:300), each then logging a BURST of ~20 in-slab candidate slots, so
        # a 100-evt sample sees 0 ~2/3 of the time (mean is carried by rare bursts).  We therefore NEVER size
        # the pion buffer from its own count: max() lets the RELIABLE nucleon cap (tail-insensitive, ~600
        # counts/100ev) cover the pion buffer too.  Empirically pion slots <= nucleon slots always; max() is
        # symmetric so it self-corrects if a config ever inverts that, and the loud guard backstops either way.
        def _cap(rec, n):
            def b(g):
                return max(64, math.ceil(int(g) / max(n, 1) * CHUNK * _MARGIN))
            t = max(b(rec["gc_p"]), b(rec["gc_n"]))
            return (t, t)
        CAPS_qe, CAPS_res = _cap(rq, nqc), _cap(rr, nrc)
        log(f"flat FSI caps auto-sized on {ncal} evts (x{_MARGIN}, symmetric max): "
            f"qe rate=({int(rq['gc_p'])/nqc:.2f},{int(rq['gc_n'])/nqc:.2f})/ev -> {CAPS_qe}   "
            f"res rate=({int(rr['gc_p'])/nrc:.2f},{int(rr['gc_n'])/nrc:.2f})/ev -> {CAPS_res}")

    def compact_fs(nt):
        al = np.asarray(nt["alive"]); chg = np.asarray(nt["charge"]); spc = np.asarray(nt["species"])
        p4 = np.asarray(nt["p4"]).astype(np.float64)
        pid = np.where(spc == 0, np.array([211, 111, -211])[np.clip(chg, 0, 2)], np.where(chg == 1, 2212, 2112))
        mom = np.sqrt(np.nan_to_num(p4[:, :, 1:] ** 2, posinf=np.inf).sum(2))
        good = al & np.isfinite(p4).all(2) & (mom < PMAX)
        cnt = good.sum(1).astype(np.int64); off = np.concatenate([[0], np.cumsum(cnt)]); m = good.reshape(-1)
        return off, pid.reshape(-1)[m].astype(np.int32), chg.reshape(-1)[m].astype(np.int32), \
            p4.reshape(-1, 4)[m].astype(np.float32), int((al & ~good).sum())

    def merge_off(a, b):
        return np.concatenate([a, a[-1] + b[1:]]).astype(np.int64)

    manifest = dict(n_chunks=n_chunks, chunk=CHUNK, n_total=CHUNK * n_chunks,
                    caps_qe=list(CAPS_qe), caps_res=list(CAPS_res))
    for c in range(n_chunks):
        qe = qe_xsec.sample_importance(CHUNK, seed=SEED0 + c, sf=sf, n_neutron=n_neutron)
        qw = np.asarray(qe["w"]) / CHUNK
        res = res_xsec.generate(CHUNK, seed=SEED0 + c, return_events=True, sf_n=sf, sf_p=sf_p,
                                n_neutron=n_neutron, n_proton=n_proton)["events"]; rw = np.asarray(res["w"])
        HV, SF = build_hv_sf(qe, res, sf, with_pw=False)
        kq, kr = jax.random.split(jax.random.PRNGKey(1000 + c), 2)
        nq = len(qw); nr = len(rw)
        ptq, ntq, _o, _cq, recq = CF.cascade_nucleus(
            jnp.zeros((nq, 4)), jnp.asarray(qe["p_out"]), jnp.zeros(nq, jnp.int32),
            jnp.full(nq, 2112, jnp.int32), jnp.full(nq, 2212, jnp.int32), POOL(seed=2), kq, channel="qe", rec_caps=CAPS_qe)
        ptr, ntr, _o2, _cr, recr = CF.cascade_nucleus(
            jnp.asarray(res["p_pi"]), jnp.asarray(res["p_N"]), jnp.asarray(res["ppid"]).astype(jnp.int32),
            jnp.asarray(res["ipid"]).astype(jnp.int32), jnp.asarray(res["Npid"]).astype(jnp.int32), POOL(seed=1), kr, channel="res", rec_caps=CAPS_res)
        log(f"chunk {c+1}/{n_chunks}: cascades done (nq={nq}, nr={nr})")

        # HARD-VERTEX records, identity-padded on the opposite channel (QE block then RES block)
        def hv_qeonly(rec):    # qe record (a,b,c,Q2) for QE rows, identity for RES rows
            return [np.concatenate([np.asarray(rec[i], np.float32), _idma(nr)[i]]) for i in range(4)]
        def hv_resonly(rec):   # identity for QE rows, res record for RES rows
            return [np.concatenate([_idma(nq)[i], np.asarray(rec[i], np.float32)]) for i in range(4)]
        qe_ma = [np.concatenate([np.asarray(HV["qe_ma"][i], np.float32), _idma(nr)[i]]) for i in range(4)]
        res_ma = [np.concatenate([_idma(nq)[i], np.asarray(HV["res_ma"][i], np.float32)]) for i in range(4)]
        hv = dict(qe_ma=qe_ma, res_ma=res_ma, qe_vec=hv_qeonly(HV["qe_vec"]),
                  qe_gmp=hv_qeonly(HV["qe_gmp"]), qe_gmn=hv_qeonly(HV["qe_gmn"]),
                  qe_gep=hv_qeonly(HV["qe_gep"]), qe_gen=hv_qeonly(HV["qe_gen"]),
                  res_pp=hv_resonly(HV["res_pp"]),
                  res_delta=hv_resonly(HV["res_delta"]))    # P33 Delta(1232) strength record
        hv_save = {f"hv_{name}_{abc}": comp[i].astype(np.float32)
                   for name, comp in hv.items() for i, abc in enumerate(["a", "b", "c", "Q2"][:len(comp)])}

        # FSI record: concatenate recq (QE) + recr (RES) per field, then COMPACT to the ragged layout.
        # The dense (n, K) record is ~97% padding (pion occupancy 2.3/96, nucleon 10.6/64); storing it
        # dense costs ~8.4 GB at 1.87M events.  compact_fsi_record drops the padding into flat (M,) slot
        # arrays + a per-slot event index -- the same ragged layout the bank already uses for the final
        # state (fs_* + fs_off).  Physics-identical (only padding is removed), ~4x smaller than the OLD
        # bank, and every Jacobian jvp then touches 40x fewer slots.
        if _flat:
            # recq/recr are already-ragged FLAT records -> compact (trim) each, concat, offset the RES
            # per-slot event index by nq (QE block is events 0..nq-1, RES block nq..nq+nr-1).
            from adonis.fsi.cascade import _P_SLOT, _N_SLOT
            cq = CF.compact_fsi_record(dict(recq)); cr = CF.compact_fsi_record(dict(recr))
            flat = {f: np.concatenate([cq[f], cr[f]]) for f in _P_SLOT + _N_SLOT}
            flat["p_eidx"] = np.concatenate([cq["p_eidx"], cr["p_eidx"] + nq])
            flat["n_eidx"] = np.concatenate([cq["n_eidx"], cr["n_eidx"] + nq])
        else:
            def fcat(field):
                return np.concatenate([np.asarray(recq[field]), np.asarray(recr[field])])
            flat = CF.compact_fsi_record({f: fcat(f) for f in FSI_F})
        fsi_save = {"f_p_eidx": flat["p_eidx"].astype(np.int32),
                    "f_n_eidx": flat["n_eidx"].astype(np.int32)}
        for field, arr in flat.items():
            if field.endswith("_eidx"):
                continue
            dt = (np.int8 if field in ("bc", "iso")
                  else (bool if field in ("hh", "inel", "swap", "pi_hh") else np.float32))
            fsi_save[f"f_{field}"] = arr.astype(dt)
        _budget = ((CAPS_qe[0] + CAPS_qe[1] + CAPS_res[0] + CAPS_res[1]) if _flat
                   else (nq + nr) * (CAPS_qe[0] + CAPS_qe[1]))
        log(f"chunk {c+1}: FSI record {'streamed(flat)' if _flat else 'compacted(dense)'} -> pion "
            f"{len(flat['p_eidx'])} slots, nucleon {len(flat['n_eidx'])} slots "
            f"({'flat budget' if _flat else 'dense would be'} {_budget})")

        offq, fpq, fcq, fp4q, dq = compact_fs(ntq[0]); offr, fpr, fcr, fp4r, dr = compact_fs(ntr[0])
        if dq + dr:
            log(f"chunk {c+1}: dropped {dq+dr} non-physical final-state particles")
        np.savez(f"{outdir}/chunk_{c:03d}.npz",
                 channel=np.concatenate([np.zeros(nq, np.int8), np.ones(nr, np.int8)]),
                 prim_pi_pid=np.concatenate([np.asarray(ptq["pid"]), np.asarray(ptr["pid"])]).astype(np.int32),
                 w0=np.concatenate([qw, rw]),
                 k_nu=np.concatenate([np.asarray(qe["k_nu"]), np.asarray(res["k_nu"])]).astype(np.float32),
                 p_struck=np.concatenate([np.asarray(qe["p_struck"]), np.asarray(res["p_struck"])]).astype(np.float32),
                 k_mu=np.concatenate([np.asarray(qe["k_mu"]), np.asarray(res["k_mu"])]).astype(np.float32),
                 fs_off=merge_off(offq, offr), fs_pid=np.concatenate([fpq, fpr]),
                 fs_chg=np.concatenate([fcq, fcr]), fs_p4=np.concatenate([fp4q, fp4r]),
                 # raw RES vertex kinematics (RES rows real, QE rows zero-padded) -> ANY future RES
                 # hard-vertex knob's amps2 record is buildable at load with NO further regen:
                 res_p_N=np.concatenate([np.zeros((nq, 4), np.float32), np.asarray(res["p_N"], np.float32)]),
                 res_p_pi=np.concatenate([np.zeros((nq, 4), np.float32), np.asarray(res["p_pi"], np.float32)]),
                 res_ipid=np.concatenate([np.zeros(nq, np.int32), np.asarray(res["ipid"], np.int32)]),
                 res_ppid=np.concatenate([np.zeros(nq, np.int32), np.asarray(res["ppid"], np.int32)]),
                 **hv_save, **fsi_save)
        del qe, res, HV, SF, recq, recr, ntq, ntr; gc.collect()
        log(f"chunk {c+1}/{n_chunks}: written")
    with open(f"{outdir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE: full-record bank in {outdir}/ ({n_chunks} chunks) + manifest.json")


if __name__ == "__main__":
    run()
