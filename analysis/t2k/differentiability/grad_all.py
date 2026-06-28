"""High-statistics driver for the knob Jacobian figures (1D arrow grids + 2D heatmap).

Two levels of streaming keep peak memory ~ ONE chunk regardless of total statistics:
  * primary events are CHUNKED (sample a chunk -> walk it -> accumulate the Jacobian -> free), because the
    proposal itself does not fit at 1e6 events.  Chunks are independent (different seed) so the Jacobian
    (and h0) sum linearly across them.
  * within a chunk, the NREP cascade replicas are STREAMED the same way.
Total primary events = CHUNK * n_chunks; NOTE only the FSI knobs gain from NREP (extra cascade
realizations of the SAME primary events) -- the hard-vertex/SF/norm knobs are limited by the primary count.

The heavy result (h0 + per-bin Jacobian J + edges + labels) is CACHED to a persistent dir so plotting can
be reworked WITHOUT re-running:
    CC0PI_N=1000000 CHUNK=80000 NREP=1 python analysis/t2k/differentiability/grad_all.py   # compute + cache + plot
    python analysis/t2k/differentiability/grad_all.py --replot                              # re-render from cache only

Cache: output/jac_cache/{jac_grid_dat,jac_grid_dpt,jac2d}.npz  (also mirrored to /tmp/adonis_tune_runs).
"""
import os, sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

NREP = int(os.environ.get("NREP", "4"))
CACHE = "output/jac_cache"
PDF_1D = "output/figures/cc0pi_jacobian_arrows_all.pdf"


def _render(GA, G2):
    """(Re)build all figures from the cached npz -- the plotting-only path."""
    npzs = []
    for o in ("dat", "dpt"):
        d = np.load(f"{CACHE}/jac_grid_{o}.npz", allow_pickle=True)
        GA.make_grid(o, d["edges"], d["h0"], d["J"], list(d["labels"]))
        npzs.append(f"{CACHE}/jac_grid_{o}.npz")
    GA.build_pdf(PDF_1D, npzs)
    G2.make_figure(f"{CACHE}/jac2d.npz")


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _replot = "--replot" in sys.argv
    sys.argv = [sys.argv[0], "dpt"]                          # tune reads OBS at import; we bin BOTH explicitly
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability import grad_arrows as GA
    from analysis.t2k.differentiability import grad_2d as G2
    from analysis.t2k.differentiability.full_knobs import (nominal_knobs, build_hv_sf, model_hist_full,
                                                           _hv_qe, _hv_res, _fsi)
    from adonis.analysis.sf_reweight import sf_reweight
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction
    from adonis.xsec import qe_xsec, res_xsec

    if _replot:
        _render(GA, G2); return

    N_TOTAL = int(os.environ.get("CC0PI_N", "120000"))
    CHUNK = min(int(os.environ.get("CHUNK", str(N_TOTAL))), N_TOTAL)
    n_chunks = int(np.ceil(N_TOTAL / CHUNK))
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    NOM = nominal_knobs()
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    log(f"grad_all: N_total={N_TOTAL} in {n_chunks} chunk(s) of {CHUNK}, NREP={NREP}  "
        f"(primary events={CHUNK*n_chunks}, cascades={CHUNK*n_chunks*NREP})")

    SP = GA._specs(NOM)
    tup_names = sorted({n for n, idx, _, _ in SP if idx is not None})
    labels = [s[2] for s in SP]; p0 = jnp.asarray([s[3] for s in SP]); nk = len(labels)

    def assemble(p):
        k = dict(NOM); tup = {n: list(NOM[n]) for n in tup_names}
        for i, (name, idx, _, _) in enumerate(SP):
            (k.__setitem__(name, p[i]) if idx is None else tup[name].__setitem__(idx, p[i]))
        for n in tup_names:
            k[n] = tuple(tup[n])
        return k

    OBS1D = {o: GA._obs_binning(T, o) for o in ("dat", "dpt")}
    edpt, edat = G2._binning("dpt"), G2._binning("dat")
    n1, n2 = len(edpt) - 1, len(edat) - 1
    ej1, ej2 = jnp.asarray(edpt), jnp.asarray(edat)
    area = np.outer(np.diff(edpt), np.diff(edat)); CONV2D = 1e-33 / 12.0 * 1e38 * 1000.0

    def bin2d(W):
        qd, qa = T._dpt(W["q_kmu"], W["q_lead"]), T._dat(W["q_kmu"], W["q_lead"])
        rd, ra = T._dpt(W["r_kmu"], W["r_lead"]), T._dat(W["r_kmu"], W["r_lead"])
        qi = jnp.clip(jnp.searchsorted(ej1, qd) - 1, 0, n1 - 1); qj = jnp.clip(jnp.searchsorted(ej2, qa) - 1, 0, n2 - 1)
        ri = jnp.clip(jnp.searchsorted(ej1, rd) - 1, 0, n1 - 1); rj = jnp.clip(jnp.searchsorted(ej2, ra) - 1, 0, n2 - 1)
        return dict(q_flat=qi * n2 + qj, q_keep=T._sel(W["q_kmu"], W["q_lead"]), q_w0=W["q_w0"], q_rec=W["q_rec"],
                    r_flat=ri * n2 + rj, r_keep=T._sel(W["r_kmu"], W["r_lead"]), r_w0=W["r_w0"], r_rec=W["r_rec"])

    def hist1d(p, R, edges, conv):
        return model_hist_full(assemble(p), R, _HV, _SF, edges, conv)

    def hist2d(p, R2):
        k = assemble(p)
        qsf = sf_reweight(_SF["grids"], _SF["qe_pmag"], _SF["qe_erem"], kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        rsf = sf_reweight(_SF["grids"], _SF["res_pmag"], _SF["res_erem"], kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        qw_ = R2["q_w0"] * k["qe_norm"] * _hv_qe(k, _HV) * _fsi(R2["q_rec"], k) * qsf
        rw_ = R2["r_w0"] * k["res_norm"] * _hv_res(k, _HV) * _fsi(R2["r_rec"], k) * rsf
        return (jax.ops.segment_sum(qw_ * R2["q_keep"], R2["q_flat"], num_segments=n1 * n2)
                + jax.ops.segment_sum(rw_ * R2["r_keep"], R2["r_flat"], num_segments=n1 * n2))
    jac1d = jax.jacrev(hist1d, argnums=0); jac2d = jax.jacrev(hist2d, argnums=0)

    h0_1d = {o: np.zeros(n1) for o in OBS1D}; J_1d = {o: np.zeros((nk, n1)) for o in OBS1D}
    h0_2d = np.zeros(n1 * n2); J_2dflat = np.zeros((n1 * n2, nk))
    _HV = _SF = None
    niter = n_chunks * NREP
    for c in range(n_chunks):
        qe = qe_xsec.sample_importance(CHUNK, seed=c)
        qw = np.asarray(qe["w"]) / CHUNK
        res = res_xsec.generate(CHUNK, seed=c, return_events=True)["events"]
        rw = np.asarray(res["w"])
        _HV, _SF = build_hv_sf(qe, res, sf, with_pw=False)
        log(f"chunk {c+1}/{n_chunks}: proposal + HV/SF built")
        for r in range(NREP):
            W = T.build_walk(jax.random.PRNGKey(1000 * c + r), qe, qw, res, rw)
            for o, (edges, conv, ofn) in OBS1D.items():
                R = T.bin_walk(W, edges=edges, obs=ofn)
                h0_1d[o] += np.asarray(hist1d(p0, R, jnp.asarray(edges), conv))
                J_1d[o] += np.asarray(jac1d(p0, R, jnp.asarray(edges), conv)).T
                del R
            R2 = bin2d(W)
            h0_2d += np.asarray(hist2d(p0, R2)); J_2dflat += np.asarray(jac2d(p0, R2))
            del W, R2; gc.collect()
            log(f"  chunk {c+1} replica {r+1}/{NREP} accumulated")
        del qe, res, _HV, _SF; gc.collect()
    for o in OBS1D:
        h0_1d[o] /= niter; J_1d[o] /= niter
    h0_2d /= niter; J_2dflat /= niter

    os.makedirs(CACHE, exist_ok=True); os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    meta = dict(N_total=CHUNK * n_chunks, n_chunks=n_chunks, chunk=CHUNK, nrep=NREP)
    for o, (edges, conv, ofn) in OBS1D.items():
        for d in (CACHE, "/tmp/adonis_tune_runs"):
            np.savez(f"{d}/jac_grid_{o}.npz", obs=o, edges=edges, h0=h0_1d[o], J=J_1d[o],
                     labels=np.array(labels), **meta)
    norm2d = (h0_2d.reshape(n1, n2) / area) * CONV2D
    J2d = (J_2dflat.T.reshape(nk, n1, n2) / area[None]) * CONV2D
    for d in (CACHE, "/tmp/adonis_tune_runs"):
        np.savez(f"{d}/jac2d.npz", edpt=edpt, edat=edat, norm2d=norm2d, J2d=J2d, labels=np.array(labels), **meta)
    log(f"cached -> {CACHE}/  (jac_grid_dat, jac_grid_dpt, jac2d).npz")
    _render(GA, G2)
    log("figures rendered")


if __name__ == "__main__":
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    run()
