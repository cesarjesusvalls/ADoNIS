"""Two follow-up diagnostics on the full17 gradient stage (run AFTER the fits; needs the bank).

(1) Eb_shift per-bin AD-vs-FD eps-scan.  The full17 gate fails ONLY for Eb_shift (3.3e-2; all 16 other
    knobs <=6e-6).  Hypothesis: with the faithful LINEAR-in-E interpolant, w(Eb) is piecewise-linear in Eb
    with kinks where E-Eb crosses an E-grid node (and where S hits the >=0 clamp); central FD across a kink
    averages two slopes, AD returns the exact local one.  If true, the AD-FD gap per bin shrinks as eps
    shrinks (fewer events with a kink inside the bracket) -- scan eps = 1e-3 / 1e-4 / 1e-5.

(2) Pion-FSI null direction.  The normalized total Fisher has an exactly-zero eigenvalue (lam ~ -9e-17)
    along ~ +0.77*sabs +0.47*s_piN_cex +0.44*s_piN_elastic.  Test whether J.v vanishes PER EVENT (structural
    identity of the kind-1 pion reweight at nominal) or only per bin (accidental cancellation in the
    binning).  v is mapped back from the normalized basis: v_knob = v_norm / sqrt(diag F).

    python -u scripts/fd_gate_diagnose.py
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.argv = [sys.argv[0], "--set", "full17"]
from scripts.info_content import (SPEC, PNAMES, THETA_NOM, NPAR, knobs_of, build_datasets, bin_w0, BANKDIR, NPZ)
from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs

K_EB = PNAMES.index("Eb_shift")


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    CK = dict(np.load(NPZ.replace(".npz", "_grad.npz"), allow_pickle=True))
    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids()
    nominal = nominal_knobs()
    ds = build_datasets(B)
    log("bank + datasets ready; checkpoint loaded")

    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nominal), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda tang, JB: jax.jvp(lambda th: wf(th, JB), (THETA_NOM,), (tang,))[1])

    # ---- (1) Eb eps-scan ------------------------------------------------------------------------------- #
    ad_e = np.asarray(jvp_wf(jnp.zeros(NPAR).at[K_EB].set(1.0), JB))
    for eps in (1e-3, 1e-4, 1e-5):
        tp = np.array(THETA_NOM); tm = np.array(THETA_NOM); tp[K_EB] += eps; tm[K_EB] -= eps
        fd_e = (np.asarray(wf_jit(jnp.asarray(tp), JB)) - np.asarray(wf_jit(jnp.asarray(tm), JB))) / (2 * eps)
        worst = 0.0
        nkink = int(np.sum(np.abs(ad_e - fd_e) > 1e-6 * max(np.abs(ad_e).max(), 1e-300)))
        for i, d in enumerate(ds):
            Jb = bin_w0(d, ad_e); Fb = bin_w0(d, fd_e)
            sc = max(np.max(np.abs(Jb)), 1e-300)
            worst = max(worst, float(np.max(np.abs(Jb - Fb)) / sc))
        log(f"(1) Eb_shift eps={eps:.0e}: per-bin worst rel = {worst:.2e}; "
            f"events with per-event AD!=FD: {nkink} ({100*nkink/len(ad_e):.3f}%)")

    # ---- (2) pion-FSI null direction ------------------------------------------------------------------- #
    ewv = CK["fisher_eigvals"]; evv = CK["fisher_eigvecs"]; F_tot = CK["F_tot"]
    j0 = int(np.argmin(np.abs(ewv)))
    vn = evv[:, j0]
    dg = np.sqrt(np.maximum(np.diag(F_tot), 1e-300))
    v = vn / dg; v = v / np.linalg.norm(v)                     # knob-space null direction
    comp = ", ".join(f"{v[i]:+.3f}*{PNAMES[i]}" for i in np.argsort(-np.abs(v))[:5])
    log(f"(2) null direction (lam={ewv[j0]:.2e}): {comp}")
    dv_e = np.asarray(jvp_wf(jnp.asarray(v), JB))              # per-event J.v
    # compare to the typical per-event gradient magnitude along the SAME knobs
    ref = np.zeros_like(dv_e)
    for i in np.argsort(-np.abs(v))[:3]:
        ref += np.abs(v[i]) * np.abs(np.asarray(jvp_wf(jnp.zeros(NPAR).at[int(i)].set(1.0), JB)))
    m = ref > 0
    log(f"(2) per-event |J.v|: max = {np.abs(dv_e).max():.3e}; "
        f"max |J.v|/ref (events with ref>0: {m.sum()}) = {np.max(np.abs(dv_e[m])/ref[m]):.3e}")
    for i, d in enumerate(ds):
        bv = bin_w0(d, dv_e)
        log(f"(2)   {d['name']:12s} per-bin |J.v| max = {np.max(np.abs(bv)):.3e}  "
            f"(vs |J| scale {np.max(np.abs(CK[f'J_{i}'])):.3e})")


if __name__ == "__main__":
    main()
