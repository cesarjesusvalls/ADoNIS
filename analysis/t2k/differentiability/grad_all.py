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
    walks = [T.build_walk(jax.random.PRNGKey(50 + i), qe, qw, res, rw) for i in range(NREP)]
    log(f"{NREP} OBSERVABLE-INDEPENDENT walk replicas built (reused for 1D + 2D)")

    SP = GA._specs(NOM)
    tup_names = sorted({n for n, idx, _, _ in SP if idx is not None})
    labels = [s[2] for s in SP]; p0 = jnp.asarray([s[3] for s in SP])

    def assemble(p):
        k = dict(NOM); tup = {n: list(NOM[n]) for n in tup_names}
        for i, (name, idx, _, _) in enumerate(SP):
            (k.__setitem__(name, p[i]) if idx is None else tup[name].__setitem__(idx, p[i]))
        for n in tup_names:
            k[n] = tuple(tup[n])
        return k

    # ---- 1D: both observables from the same walks ------------------------------------------------- #
    npzs = []
    for obs in ("dat", "dpt"):
        edges, conv, ofn = GA._obs_binning(T, obs)
        bank = [T.bin_walk(W, edges=edges, obs=ofn) for W in walks]

        def hist_mean(p, bank=bank, edges=edges, conv=conv):
            kk = assemble(p)
            return jnp.mean(jnp.stack([model_hist_full(kk, R, HV, SF, edges, conv) for R in bank]), axis=0)

        h0 = np.asarray(hist_mean(p0)); Js = np.asarray(jax.jacrev(hist_mean)(p0)).T
        np.savez(f"/tmp/adonis_tune_runs/jac_grid_{obs}.npz", obs=obs, edges=edges, h0=h0, J=Js,
                 labels=np.array(labels))
        npzs.append(f"/tmp/adonis_tune_runs/jac_grid_{obs}.npz")
        GA.make_grid(obs, edges, h0, Js, labels)
        log(f"1D {obs} done")
    GA.build_pdf("output/figures/cc0pi_jacobian_arrows_all.pdf", npzs)

    # ---- 2D: (dpt,dat) plane from the same walks ------------------------------------------------- #
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
    banks = [bin2d(W) for W in walks]
    qsf_args = (SF["grids"], SF["qe_pmag"], SF["qe_erem"]); rsf_args = (SF["grids"], SF["res_pmag"], SF["res_erem"])

    def hist2d_flat(p):
        k = assemble(p)
        qsf = sf_reweight(*qsf_args, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        rsf = sf_reweight(*rsf_args, kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"], sf_norm=k["sf_norm"], src_tail=k["src_tail"])
        hvq, hvr = _hv_qe(k, HV), _hv_res(k, HV)
        acc = jnp.zeros(n1 * n2)
        for R in banks:
            q_w = R["q_w0"] * k["qe_norm"] * hvq * _fsi(R["q_rec"], k) * qsf
            r_w = R["r_w0"] * k["res_norm"] * hvr * _fsi(R["r_rec"], k) * rsf
            acc = acc + (jax.ops.segment_sum(q_w * R["q_keep"], R["q_flat"], num_segments=n1 * n2)
                         + jax.ops.segment_sum(r_w * R["r_keep"], R["r_flat"], num_segments=n1 * n2))
        return acc / len(banks)

    h0 = np.asarray(hist2d_flat(p0)).reshape(n1, n2)
    Jflat = np.asarray(jax.jacrev(hist2d_flat)(p0))
    norm2d = (h0 / area) * CONV2D
    J2d = (Jflat.T.reshape(len(labels), n1, n2) / area[None]) * CONV2D
    np.savez("/tmp/adonis_tune_runs/jac2d.npz", edpt=edpt, edat=edat, norm2d=norm2d, J2d=J2d, labels=np.array(labels))
    G2.make_figure("/tmp/adonis_tune_runs/jac2d.npz")
    log("2D done")


if __name__ == "__main__":
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    run()
