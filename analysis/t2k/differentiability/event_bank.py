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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
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
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, build_hv_sf
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction
    from adonis.xsec import qe_xsec, res_xsec
    import adonis.fsi.cascade_full as CF

    N_TOTAL = int(os.environ.get("CC0PI_N", "100000"))
    CHUNK = min(int(os.environ.get("CHUNK", str(N_TOTAL))), N_TOTAL)
    n_chunks = int(np.ceil(N_TOTAL / CHUNK))
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    NOM = nominal_knobs()
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    log(f"event_bank (full records): N={N_TOTAL} in {n_chunks} chunk(s) of {CHUNK} -> {outdir}")
    POOL = T.POOLCFG; CAPS = T.REC_CAPS
    FSI_F = ("bc", "sa", "ss_el", "ss", "si", "nh", "hh", "a", "iso", "finel", "inel", "swap", "ns")

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

    manifest = dict(n_chunks=n_chunks, chunk=CHUNK, n_total=CHUNK * n_chunks, caps=list(CAPS))
    for c in range(n_chunks):
        qe = qe_xsec.sample_importance(CHUNK, seed=c); qw = np.asarray(qe["w"]) / CHUNK
        res = res_xsec.generate(CHUNK, seed=c, return_events=True)["events"]; rw = np.asarray(res["w"])
        HV, SF = build_hv_sf(qe, res, sf, with_pw=False)
        kq, kr = jax.random.split(jax.random.PRNGKey(1000 + c), 2)
        nq = len(qw); nr = len(rw)
        ptq, ntq, _o, _cq, recq = CF.cascade_nucleus(
            jnp.zeros((nq, 4)), jnp.asarray(qe["p_out"]), jnp.zeros(nq, jnp.int32),
            jnp.full(nq, 2112, jnp.int32), jnp.full(nq, 2212, jnp.int32), POOL(seed=2), kq, channel="qe", rec_caps=CAPS)
        ptr, ntr, _o2, _cr, recr = CF.cascade_nucleus(
            jnp.asarray(res["p_pi"]), jnp.asarray(res["p_N"]), jnp.asarray(res["ppid"]).astype(jnp.int32),
            jnp.asarray(res["ipid"]).astype(jnp.int32), jnp.asarray(res["Npid"]).astype(jnp.int32), POOL(seed=1), kr, channel="res", rec_caps=CAPS)
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
                  res_pp=hv_resonly(HV["res_pp"]))
        hv_save = {f"hv_{name}_{abc}": comp[i].astype(np.float32)
                   for name, comp in hv.items() for i, abc in enumerate(["a", "b", "c", "Q2"][:len(comp)])}

        # FSI record: concatenate recq (QE) + recr (RES) per field
        def fcat(field):
            a = np.asarray(recq[field]); b = np.asarray(recr[field])
            return np.concatenate([a, b])
        fsi_save = {}
        for field in FSI_F:
            arr = fcat(field)
            dt = (np.int16 if field in ("nh", "ns") else (np.int8 if field in ("bc", "iso")
                  else (bool if field in ("hh", "inel", "swap") else np.float32)))
            fsi_save[f"f_{field}"] = arr.astype(dt)

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
                 **hv_save, **fsi_save)
        del qe, res, HV, SF, recq, recr, ntq, ntr; gc.collect()
        log(f"chunk {c+1}/{n_chunks}: written")
    with open(f"{outdir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE: full-record bank in {outdir}/ ({n_chunks} chunks) + manifest.json")


if __name__ == "__main__":
    run()
