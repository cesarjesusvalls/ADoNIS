"""Validation gates for the CC0pi LLH-surface machinery (run before mapping any surface).

  A. slicing bit-exactness : sum_{signal} weight_jit(full)  ==  sum weight_jit(sliced)  (identity)
  B. nominal footprint     : model_vec(nominal, the DIFFERENTIABLE nominal: Eb=1e-2, f_NN_cex=0.5) vs the
                             bank_plot w0-forward (cc0pi_data_chi2, bare w0, NO nominal reweights).  These
                             are legitimately different objects; the difference is the nominal SF/FSI
                             reweight footprint -- INFORMATIONAL, expected small in aggregate.
  C. AD Hessian == FD      : jax.hessian(chi2_pair) vs central finite differences at a test point
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import cc0pi_llh_lib as L
from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    JBs, grids, aux = L.load_signal_bank()
    NOM = nominal_knobs()

    # ---- Gate A: slicing is a pure identity ---------------------------------------------------- #
    B = aux["B"]; mask = aux["mask"]
    JBfull = BR.to_jax(B)
    wfull = np.asarray(BR.weight_jit(JBfull, NOM, grids))
    wslice = np.asarray(BR.weight_jit(JBs, NOM, grids))
    sA_full = float(wfull[mask].sum()); sA_slice = float(wslice.sum())
    relA = abs(sA_full - sA_slice) / abs(sA_full)
    log(f"GATE A slicing: sum(full[mask])={sA_full:.8e}  sum(slice)={sA_slice:.8e}  rel={relA:.2e}  "
        f"{'PASS' if relA < 1e-12 else 'FAIL'}")

    # ---- Gate B: nominal model vs bank_plot w0-forward ----------------------------------------- #
    specs = [L.obs_binning(aux, o) for o in ("dpt", "dat")]
    C = L.make_chi2(JBs, grids, specs)
    mvec = np.asarray(C["model_vec"](NOM))
    # bank_plot w0-forward (cc0pi_data_chi2.adonis_nominal), per obs
    lead = aux["lead"]; w0 = B["w0"].astype(np.float64)
    off = 0
    for s in specs:
        obs = s["obs"]; edges = s["edges"]; conv = s["conv"]; nb = s["nb"]
        val = np.asarray(BP.dpt(B, lead) if obs == "dpt" else BP.dat(B, lead))
        h, _ = np.histogram(val[mask], bins=edges, weights=w0[mask])
        fwd = h / np.diff(edges) * conv
        mv = mvec[off:off + nb]; off += nb
        rel = np.max(np.abs(mv - fwd) / np.clip(np.abs(fwd), 1e-30, None))
        aggr = abs(mv.sum() - fwd.sum()) / abs(fwd.sum())
        log(f"GATE B footprint[{obs}]: max per-bin |diff|={rel:.2e}  aggregate |diff|={aggr:.2e}  "
            f"(nominal SF Eb=1e-2 + FSI f_NN_cex=0.5 reweight; INFORMATIONAL)")

    # ---- covariance conditioning + nominal chi2 ------------------------------------------------ #
    for nm, cond, nf in zip(C["obs_names"], C["conds"], C["nfloored"]):
        log(f"cov[{nm}]: cond={cond:.3e}  eigs_floored={nf}")
    c2a = float(C["chi2_abs"](NOM)); c2s = float(C["chi2_shape"](NOM))
    log(f"nominal chi2/ndf: abs={c2a/C['ndf']:.3f}  shape={c2s/C['ndf']:.3f}  (ndf={C['ndf']})")

    # ---- Gate C: AD Hessian vs FD for a scalar pair -------------------------------------------- #
    names = ["M_A_qe", "axial_strength"]
    assemble, _ = L.assembler(names)
    chi2_pair = lambda p: C["chi2_abs"](assemble(p))    # NOT jitted: jitting would bake JBs -> NaN (see lib)
    p0 = jnp.array([1.05, 0.95])
    Had = np.asarray(jax.hessian(chi2_pair)(p0))
    eps = 1e-4
    Hfd = np.zeros((2, 2))
    f = lambda p: float(chi2_pair(jnp.asarray(p)))
    p0n = np.asarray(p0)
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2); ei[i] = eps; ej = np.zeros(2); ej[j] = eps
            Hfd[i, j] = (f(p0n + ei + ej) - f(p0n + ei - ej) - f(p0n - ei + ej) + f(p0n - ei - ej)) / (4 * eps * eps)
    relC = np.max(np.abs(Had - Hfd) / np.clip(np.abs(Hfd), 1e-6, None))
    log(f"GATE C AD-Hessian vs FD: relmax={relC:.2e}  {'PASS' if relC < 1e-3 else 'FAIL'}")
    log(f"  H_AD=\n{Had}\n  H_FD=\n{Hfd}")


if __name__ == "__main__":
    main()
