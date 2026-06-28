"""High-statistics driver: build the frozen proposal + pool walk bank ONCE, then emit BOTH the 1D arrow
grids (dpt & dat -> PDF, via grad_arrows) AND the 2D (dpt,dat) sensitivity heatmap (via grad_2d), reusing
the SAME walk for everything (one expensive cascade build instead of two).

  CC0PI_N=120000 python analysis/t2k/differentiability/grad_all.py

Outputs (same filenames as the standalone scripts, so --plot-only re-render still works):
  output/figures/cc0pi_jacobian_arrows_grid_{dat,dpt}.png + cc0pi_jacobian_arrows_all.pdf   (1D)
  output/figures/cc0pi_jacobian_2d.{png,pdf}                                                (2D)
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

NREP = int(os.environ.get("NREP", "4"))


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability import grad_arrows as GA
    from analysis.t2k.differentiability import grad_2d as G2
    from analysis.t2k.differentiability.full_knobs import (nominal_knobs, build_hv_sf, model_hist_full,
                                                           _hv_qe, _hv_res, _fsi)
    from adonis.analysis.sf_reweight import sf_reweight
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction

    T.NQE = T.NRES = int(os.environ.get("CC0PI_N", "120000"))
    NOM = nominal_knobs()
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log(f"grad_all (1D grids + 2D heatmap), all knobs minus pw_norm.  N={T.NQE} NREP={NREP}")

    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    HV, SF = build_hv_sf(qe, res, sf, with_pw=False); log("hard-vertex amps2 records (no pw) + SF grids built")

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

    # binning constants (observable-independent build; replicas STREAMED -> peak mem = ONE replica)
    OBS1D = {o: GA._obs_binning(T, o) for o in ("dat", "dpt")}        # o -> (edges, conv, obs_fn)
    edpt, edat = G2._binning("dpt"), G2._binning("dat")
    n1, n2 = len(edpt) - 1, len(edat) - 1
    ej1, ej2 = jnp.asarray(edpt), jnp.asarray(edat)
    area = np.outer(np.diff(edpt), np.diff(edat)); CONV2D = 1e-33 / 12.0 * 1e38 * 1000.0
    qsf_args = (SF["grids"], SF["qe_pmag"], SF["qe_erem"]); rsf_args = (SF["grids"], SF["res_pmag"], SF["res_erem"])

    def bin2d(W):
        qd, qa = T._dpt(W["q_kmu"], W["q_lead"]), T._dat(W["q_kmu"], W["q_lead"])
        rd, ra = T._dpt(W["r_kmu"], W["r_lead"]), T._dat(W["r_kmu"], W["r_lead"])
        qi = jnp.clip(jnp.searchsorted(ej1, qd) - 1, 0, n1 - 1); qj = jnp.clip(jnp.searchsorted(ej2, qa) - 1, 0, n2 - 1)
        ri = jnp.clip(jnp.searchsorted(ej1, rd) - 1, 0, n1 - 1); rj = jnp.clip(jnp.searchsorted(ej2, ra) - 1, 0, n2 - 1)
        return dict(q_flat=qi * n2 + qj, q_keep=T._sel(W["q_kmu"], W["q_lead"]), q_w0=W["q_w0"], q_rec=W["q_rec"],
                    r_flat=ri * n2 + rj, r_keep=T._sel(W["r_kmu"], W["r_lead"]), r_w0=W["r_w0"], r_rec=W["r_rec"])

    # differentiated functions: knob-vector p is argnums=0; the (single-replica) record R is a plain arg
    # so JAX compiles ONCE and reuses across replicas (no recompile, no bank held resident).
    def hist1d(p, R, edges, conv):
        return model_hist_full(assemble(p), R, HV, SF, edges, conv)

    def hist2d(p, R2):
        k = assemble(p)
        qsf = sf_reweight(*qsf_args, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        rsf = sf_reweight(*rsf_args, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        qw_ = R2["q_w0"] * k["qe_norm"] * _hv_qe(k, HV) * _fsi(R2["q_rec"], k) * qsf
        rw_ = R2["r_w0"] * k["res_norm"] * _hv_res(k, HV) * _fsi(R2["r_rec"], k) * rsf
        return (jax.ops.segment_sum(qw_ * R2["q_keep"], R2["q_flat"], num_segments=n1 * n2)
                + jax.ops.segment_sum(rw_ * R2["r_keep"], R2["r_flat"], num_segments=n1 * n2))
    jac1d = jax.jacrev(hist1d, argnums=0); jac2d = jax.jacrev(hist2d, argnums=0)

    # streaming accumulators
    h0_1d = {o: np.zeros(n1) for o in OBS1D}; J_1d = {o: np.zeros((nk, n1)) for o in OBS1D}
    h0_2d = np.zeros(n1 * n2); J_2dflat = np.zeros((n1 * n2, nk))
    for i in range(NREP):
        W = T.build_walk(jax.random.PRNGKey(50 + i), qe, qw, res, rw)
        for o, (edges, conv, ofn) in OBS1D.items():
            R = T.bin_walk(W, edges=edges, obs=ofn)
            h0_1d[o] += np.asarray(hist1d(p0, R, jnp.asarray(edges), conv))
            J_1d[o] += np.asarray(jac1d(p0, R, jnp.asarray(edges), conv)).T
            del R
        R2 = bin2d(W)
        h0_2d += np.asarray(hist2d(p0, R2)); J_2dflat += np.asarray(jac2d(p0, R2))
        del W, R2
        log(f"replica {i+1}/{NREP} accumulated (streamed)")
    for o in OBS1D:
        h0_1d[o] /= NREP; J_1d[o] /= NREP
    h0_2d /= NREP; J_2dflat /= NREP

    # ---- 1D figures + PDF --------------------------------------------------------------------------- #
    npzs = []
    for o, (edges, conv, ofn) in OBS1D.items():
        np.savez(f"/tmp/adonis_tune_runs/jac_grid_{o}.npz", obs=o, edges=edges, h0=h0_1d[o], J=J_1d[o],
                 labels=np.array(labels))
        npzs.append(f"/tmp/adonis_tune_runs/jac_grid_{o}.npz")
        GA.make_grid(o, edges, h0_1d[o], J_1d[o], labels)
    GA.build_pdf("output/figures/cc0pi_jacobian_arrows_all.pdf", npzs)

    # ---- 2D figure -------------------------------------------------------------------------------- #
    norm2d = (h0_2d.reshape(n1, n2) / area) * CONV2D
    J2d = (J_2dflat.T.reshape(nk, n1, n2) / area[None]) * CONV2D
    np.savez("/tmp/adonis_tune_runs/jac2d.npz", edpt=edpt, edat=edat, norm2d=norm2d, J2d=J2d, labels=np.array(labels))
    G2.make_figure("/tmp/adonis_tune_runs/jac2d.npz")
    log("2D done")


if __name__ == "__main__":
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    run()
