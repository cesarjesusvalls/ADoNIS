"""Build a persistent DIFFERENTIABLE EVENT BANK: one chunked/streamed pass over QE+RES CC-inclusive events,
storing per event everything needed to (re)make ANY plot at plot time WITHOUT re-running:

  * kinematics   : k_nu, p_struck, k_mu (initial + lepton, 4-vectors)              [f32]
  * final state  : the FULL post-FSI hadron list (pid, charge, p4), RAGGED (CSR)   [f32]  -> any signal def
  * weight       : bare w0 (NO signal folded in; nominal cross-section weight)      [f64]
  * derivatives  : d^1,d^2,d^3 w / d theta_k^k  (gradient + DIAGONAL 2nd & 3rd)     [f64]  for 27 knobs

The full final state (every escaped nucleon + surviving pion) means signal definitions -- CC0pi, CC1pi,
N-proton multiplicity, momentum windows -- are a PLOT-TIME selection, not baked in.  The frozen-walk design
makes this exact: each event has a fixed nominal final state and theta enters only as the per-event weight,
whose 1st/2nd/3rd diagonal derivatives are stored (off-diagonal Hessian is reconstructible as Ji*Jj/w0).

Knob set + order = grad_arrows._specs(NOM) (the 27 knobs, pw_norm excluded).  Derivatives via vmap'd
nested jacfwd along each one-hot knob direction (1 compile, 27 directions x 3 orders).

  CC0PI_N=100000 CHUNK=100000 python analysis/t2k/differentiability/event_bank.py [outdir]
"""
import os, sys, time, json, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

OUTDIR = "output/event_bank"


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.argv_kept = [a for a in sys.argv[1:] if not a.startswith("-")]
    outdir = sys.argv_kept[0] if sys.argv_kept else OUTDIR
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability import grad_arrows as GA
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, build_hv_sf, _hv_qe, _hv_res, _fsi
    from adonis.analysis.sf_reweight import sf_reweight
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction
    from adonis.xsec import qe_xsec, res_xsec
    import adonis.fsi.cascade_full as CF

    N_TOTAL = int(os.environ.get("CC0PI_N", "100000"))
    CHUNK = min(int(os.environ.get("CHUNK", str(N_TOTAL))), N_TOTAL)
    n_chunks = int(np.ceil(N_TOTAL / CHUNK))
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    NOM = nominal_knobs()
    SP = GA._specs(NOM)
    tup_names = sorted({n for n, idx, _, _ in SP if idx is not None})
    labels = [s[2] for s in SP]; p0 = jnp.asarray([s[3] for s in SP]); NK = len(labels)
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    log(f"event_bank: N={N_TOTAL} in {n_chunks} chunk(s) of {CHUNK}; {NK} knobs -> {outdir}")

    def assemble(p):
        k = dict(NOM); tup = {n: list(NOM[n]) for n in tup_names}
        for i, (name, idx, _, _) in enumerate(SP):
            (k.__setitem__(name, p[i]) if idx is None else tup[name].__setitem__(idx, p[i]))
        for n in tup_names:
            k[n] = tuple(tup[n])
        return k

    def derivs(wfn):
        """(w0, D1,D2,D3) each (n, NK): diagonal 1st/2nd/3rd deriv of w wrt each knob, via nested jacfwd."""
        def along(ej):
            g = lambda t: wfn(p0 + t * ej)
            d1 = jax.jacfwd(g)(0.0)
            d2 = jax.jacfwd(jax.jacfwd(g))(0.0)
            d3 = jax.jacfwd(jax.jacfwd(jax.jacfwd(g)))(0.0)
            return d1, d2, d3
        D1, D2, D3 = jax.vmap(along)(jnp.eye(NK))          # each (NK, n)
        return np.asarray(wfn(p0)), np.asarray(D1).T, np.asarray(D2).T, np.asarray(D3).T

    POOL = T.POOLCFG; CAPS = T.REC_CAPS

    PMAX = 1e4  # MeV: physical ceiling for any final-state hadron (T2K energies ~GeV); above this is a
                # known rare cascade artifact (~0.1% of particles, inf/sentinel momenta) -> dropped here so
                # the stored bank is finite & safe for ANY signal def (acceptance rejects them anyway).

    def compact_finalstate(nt):
        """Ragged (CSR) compaction of the (n, M_out) terminal buffer -> alive, PHYSICAL particles only."""
        al = np.asarray(nt["alive"]); pid = np.asarray(nt["pid"]); chg = np.asarray(nt["charge"])
        p4 = np.asarray(nt["p4"]).astype(np.float64)
        mom = np.sqrt(np.nan_to_num(p4[:, :, 1:] ** 2, posinf=np.inf).sum(2))
        good = al & np.isfinite(p4).all(2) & (mom < PMAX)
        ndrop = int((al & ~good).sum())
        cnt = good.sum(1).astype(np.int64)
        off = np.concatenate([[0], np.cumsum(cnt)])
        m = good.reshape(-1)
        return (off, pid.reshape(-1)[m].astype(np.int32), chg.reshape(-1)[m].astype(np.int32),
                p4.reshape(-1, 4)[m].astype(np.float32), ndrop)

    manifest = dict(n_chunks=n_chunks, chunk=CHUNK, n_total=CHUNK * n_chunks, labels=labels,
                    knobs_nominal=[float(x) for x in np.asarray(p0)])
    for c in range(n_chunks):
        qe = qe_xsec.sample_importance(CHUNK, seed=c); qw = np.asarray(qe["w"]) / CHUNK
        res = res_xsec.generate(CHUNK, seed=c, return_events=True)["events"]; rw = np.asarray(res["w"])
        HV, SF = build_hv_sf(qe, res, sf, with_pw=False)
        kq, kr = jax.random.split(jax.random.PRNGKey(1000 + c), 2)
        nq = len(qw); nr = len(rw)
        # cascades (full final state in nterms[0]; kind-1 record for reweights)
        ptq, ntq, _oq, _cq, recq = CF.cascade_nucleus(
            jnp.zeros((nq, 4)), jnp.asarray(qe["p_out"]), jnp.zeros(nq, jnp.int32),
            jnp.full(nq, 2112, jnp.int32), jnp.full(nq, 2212, jnp.int32), POOL(seed=2), kq,
            channel="qe", rec_caps=CAPS)
        ptr, ntr, _orr, _cr, recr = CF.cascade_nucleus(
            jnp.asarray(res["p_pi"]), jnp.asarray(res["p_N"]), jnp.asarray(res["ppid"]).astype(jnp.int32),
            jnp.asarray(res["ipid"]).astype(jnp.int32), jnp.full(nr, 2212, jnp.int32), POOL(seed=1), kr,
            channel="res", rec_caps=CAPS)
        log(f"chunk {c+1}/{n_chunks}: cascades done (nq={nq}, nr={nr})")

        def w_qe(p):
            k = assemble(p)
            sfw = sf_reweight(SF["grids"], SF["qe_pmag"], SF["qe_erem"], kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"],
                              sf_norm=k["sf_norm"], src_tail=k["src_tail"])
            return jnp.asarray(qw) * k["qe_norm"] * _hv_qe(k, HV) * _fsi(recq, k) * sfw

        def w_res(p):
            k = assemble(p)
            sfw = sf_reweight(SF["grids"], SF["res_pmag"], SF["res_erem"], kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"],
                              sf_norm=k["sf_norm"], src_tail=k["src_tail"])
            return jnp.asarray(rw) * k["res_norm"] * _hv_res(k, HV) * _fsi(recr, k) * sfw

        w0q, D1q, D2q, D3q = derivs(w_qe)
        w0r, D1r, D2r, D3r = derivs(w_res)
        log(f"chunk {c+1}: derivatives done")

        offq, fpq, fcq, fp4q, dropq = compact_finalstate(ntq[0])
        offr, fpr, fcr, fp4r, dropr = compact_finalstate(ntr[0])
        if dropq or dropr:
            log(f"chunk {c+1}: dropped {dropq+dropr} non-physical final-state particles (|p|>1e4 MeV / non-finite)")
        np.savez(f"{outdir}/chunk_{c:03d}.npz",
                 # per-event scalars / kinematics / derivatives (QE then RES; channel 0=qe,1=res)
                 channel=np.concatenate([np.zeros(nq, np.int8), np.ones(nr, np.int8)]),
                 prim_pi_pid=np.concatenate([np.asarray(ptq["pid"]), np.asarray(ptr["pid"])]).astype(np.int32),
                 w0=np.concatenate([w0q, w0r]),
                 D1=np.concatenate([D1q, D1r]).astype(np.float64),
                 D2=np.concatenate([D2q, D2r]).astype(np.float64),
                 D3=np.concatenate([D3q, D3r]).astype(np.float64),
                 k_nu=np.concatenate([np.asarray(qe["k_nu"]), np.asarray(res["k_nu"])]).astype(np.float32),
                 p_struck=np.concatenate([np.asarray(qe["p_struck"]), np.asarray(res["p_struck"])]).astype(np.float32),
                 k_mu=np.concatenate([np.asarray(qe["k_mu"]), np.asarray(res["k_mu"])]).astype(np.float32),
                 # ragged final state, QE and RES concatenated (offsets re-based for the merged event order)
                 fs_off=_merge_offsets(offq, offr),
                 fs_pid=np.concatenate([fpq, fpr]), fs_chg=np.concatenate([fcq, fcr]),
                 fs_p4=np.concatenate([fp4q, fp4r]))
        del qe, res, HV, SF, recq, recr, ntq, ntr; gc.collect()
    with open(f"{outdir}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE: bank in {outdir}/ ({n_chunks} chunks) + manifest.json")


def _merge_offsets(offa, offb):
    """Concatenate two CSR offset arrays (events a then b) into one (len = na+nb+1)."""
    return np.concatenate([offa, offa[-1] + offb[1:]]).astype(np.int64)


if __name__ == "__main__":
    run()
