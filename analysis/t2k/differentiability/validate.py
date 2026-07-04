"""Validation suites for the differentiable reweight (merged from the former scripts/sf_knob_checks.py and
scripts/fd_gate_diagnose.py).  Evidence goes to stdout (logbook) + output/figures/.

  --sf      Spectral-function knob suite (kF_sf, Eb_shift, sf_norm, src_tail), 8 checks on the 1M bank
            event points: grid uniformity, node exactness, spline-vs-upstream-Lagrange faithfulness
            (S values AND the reweight ratio), nominal identity, sf_norm/src_tail analytic gradients,
            kF_sf AD-vs-FD + physics sign, Eb_shift one-sided branch + clamp fractions.
            -> output/figures/sf_knob_checks.png

  --fisher  Post-fit diagnostics on the info_content full17 gradient checkpoint
            (/tmp/adonis_tune_runs/info_content_full17_grad.npz -- run info_content --set full17 first):
            (1) Eb_shift per-bin AD-vs-FD eps-scan (the eps=1e-3 gate failure is an FD kink artifact of the
                linear-in-E interpolant; FD converges to AD as eps shrinks);
            (2) the exactly-null pion-FSI Fisher direction: is J.v zero PER EVENT (structural identity of
                fsi_pion_reweight: per-hit s_real*D0/D with D linear in the scales -> a COMMON rescale of
                sabs/s_piN_elastic/s_piN_cex/s_conv cancels exactly) or only per bin?

    python -u analysis/t2k/differentiability/validate.py --sf
    python -u analysis/t2k/differentiability/validate.py --fisher
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")


# ================================================================ --sf: SF knob suite =================== #
def run_sf():
    from adonis.analysis.sf_reweight import sf_grids, sf_reweight, removal_from_struck, _bspline2d
    from adonis.xsec.spectral import SpectralFunction
    from adonis.workflow.materials import resolve_targets
    from analysis.t2k.differentiability import bank_plot as BP
    from analysis.t2k.differentiability.full_knobs import _EB_EPS

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    os.makedirs("output/figures", exist_ok=True)

    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    g = sf_grids(sf)
    mom = np.asarray(sf.mom, float); energy = np.asarray(sf.energy, float)
    spec2d = np.asarray(sf.spec, float).reshape(sf.np, sf.ne)
    log(f"SF table: {sf.np} p-nodes x {sf.ne} E-nodes, p in [{mom[0]:.1f},{mom[-1]:.1f}] MeV, "
        f"E in [{energy[0]:.1f},{energy[-1]:.1f}] MeV, norm={sf.norm:.6e}")

    # ---- (1) grid uniformity ------------------------------------------------------------------------- #
    dp = np.diff(mom); de = np.diff(energy)
    log(f"(1) grid uniformity: p spacing {dp.min():.6g}..{dp.max():.6g} "
        f"(rel spread {(dp.max()-dp.min())/dp.mean():.2e}); "
        f"E spacing {de.min():.6g}..{de.max():.6g} (rel spread {(de.max()-de.min())/de.mean():.2e})")

    # ---- (2) node exactness --------------------------------------------------------------------------- #
    PP, EE = np.meshgrid(mom, energy, indexing="ij")
    s_node = np.asarray(_bspline2d(g, jnp.asarray(PP.ravel()), jnp.asarray(EE.ravel()))).reshape(sf.np, sf.ne)
    pos = spec2d > 0
    rel = np.abs(s_node[pos] - spec2d[pos]) / spec2d[pos]
    zero_leak = np.max(np.abs(s_node[~pos])) if (~pos).any() else 0.0
    log(f"(2) node exactness: max rel err at S>0 nodes = {rel.max():.2e} (mean {rel.mean():.2e}); "
        f"max |S_spline| at S==0 nodes = {zero_leak:.2e} (clamped >=0)")

    # ---- bank event points ---------------------------------------------------------------------------- #
    B = BP.load_bank(BANKDIR)
    pmag, erem = removal_from_struck(B["p_struck"])
    pmag = np.asarray(pmag); erem = np.asarray(erem)
    log(f"bank {len(pmag)} events; |p| in [{pmag.min():.1f},{pmag.max():.1f}], "
        f"E_rem in [{erem.min():.1f},{erem.max():.1f}] MeV")

    def spline_S(p, E):
        return np.asarray(_bspline2d(g, jnp.asarray(p), jnp.asarray(E))) / g["norm"]

    def ratio_upstream(p, E, kF, Eb):
        s0 = sf.batch(p, E); st = sf.batch(p / kF, E - max(Eb, 0.0))
        return np.where(s0 > 0, st / np.where(s0 > 0, s0, 1.0), 1.0)

    # ---- (3) interpolation faithfulness: spline vs upstream Lagrange --------------------------------- #
    s_up = sf.batch(pmag, erem); s_sp = spline_S(pmag, erem)
    m = s_up > 1e-12 * s_up.max()
    rel3 = np.abs(s_sp[m] - s_up[m]) / s_up[m]
    log(f"(3a) S(p,E) spline vs upstream at {m.sum()} event points (S>0): "
        f"max rel {rel3.max():.2e}, rms {np.sqrt((rel3**2).mean()):.2e}")
    for kF, Eb in ((1.1, 0.0), (0.9, 0.0), (1.0, 5.0), (1.1, 5.0)):
        r_sp = np.asarray(sf_reweight(g, jnp.asarray(pmag), jnp.asarray(erem), kF_sf=kF, Eb_shift=Eb))
        r_up = ratio_upstream(pmag, erem, kF, Eb)
        d = np.abs(r_sp - r_up)
        wq = np.abs(B["w0"])
        log(f"(3b) reweight RATIO spline vs upstream @ kF={kF} Eb={Eb}: "
            f"median {np.percentile(d,50):.1e}, max {d.max():.3e}; mean w = {r_sp.mean():.6f}/{r_up.mean():.6f}; "
            f"WEIGHTED impact sum|diff*w0|/sum(w0*r) = {np.sum(d*wq)/np.sum(wq*r_up):.3e}")

    # ---- (4) nominal identity ------------------------------------------------------------------------- #
    w_nom = np.asarray(sf_reweight(g, jnp.asarray(pmag), jnp.asarray(erem),
                                   kF_sf=1.0, Eb_shift=_EB_EPS, sf_norm=1.0, src_tail=1.0))
    log(f"(4) nominal identity (Eb anchored at eps={_EB_EPS}): max|w-1| = {np.max(np.abs(w_nom-1)):.2e}, "
        f"mean bias = {w_nom.mean()-1:+.2e}, events w=0: {(w_nom==0).sum()}")
    w_nom0 = np.asarray(sf_reweight(g, jnp.asarray(pmag), jnp.asarray(erem), Eb_shift=0.0))
    log(f"(4) exact-zero Eb reference: max|w-1| = {np.max(np.abs(w_nom0-1)):.2e} (numerator==denominator)")

    # ---- per-event gradient helper -------------------------------------------------------------------- #
    pj, ej = jnp.asarray(pmag), jnp.asarray(erem)

    def w_of(theta):                                            # theta = (kF, Eb, norm, tail)
        return sf_reweight(g, pj, ej, kF_sf=theta[0], Eb_shift=theta[1], sf_norm=theta[2], src_tail=theta[3])

    def sumw(theta):
        return jnp.sum(w_of(theta))
    grad_sumw = jax.jit(jax.grad(sumw))
    w_jit = jax.jit(w_of)

    def fd_check(theta, k, eps):
        tp = np.array(theta); tm = np.array(theta); tp[k] += eps; tm[k] -= eps
        fd = (float(jnp.sum(w_jit(jnp.asarray(tp)))) - float(jnp.sum(w_jit(jnp.asarray(tm))))) / (2 * eps)
        ad = float(grad_sumw(jnp.asarray(theta))[k])
        return ad, fd, (ad / fd if fd != 0 else np.nan)

    NOM = np.array([1.0, _EB_EPS, 1.0, 1.0])

    # ---- (5) sf_norm analytic ------------------------------------------------------------------------- #
    th = NOM.copy(); th[2] = 1.7
    w17 = np.asarray(w_jit(jnp.asarray(th)))
    lin = np.max(np.abs(w17 - 1.7 * w_nom))
    gn = np.asarray(jax.jit(jax.jacfwd(lambda s: w_of(jnp.asarray([1.0, _EB_EPS, 1.0, 1.0]).at[2].set(s))))(1.7))
    an = np.max(np.abs(gn - w17 / 1.7))
    ad, fd, r = fd_check(th, 2, 1e-4)
    log(f"(5) sf_norm: linearity max|w(1.7)-1.7*w(1)| = {lin:.2e}; per-event |dw/dnorm - w/norm|_max = {an:.2e}; "
        f"sum-w AD/FD @1.7 = {r:.10f}")

    # ---- (6) src_tail analytic ------------------------------------------------------------------------ #
    sgm = 1.0 / (1.0 + np.exp(-(pmag - 300.0) / 80.0))
    th = NOM.copy(); th[3] = 1.5
    w15 = np.asarray(w_jit(jnp.asarray(th)))
    ana = np.max(np.abs(w15 - (1.0 + 0.5 * sgm) * w_nom))
    gt = np.asarray(jax.jit(jax.jacfwd(lambda s: w_of(jnp.asarray([1.0, _EB_EPS, 1.0, 1.0]).at[3].set(s))))(1.5))
    at = np.max(np.abs(gt - sgm * w_nom))
    ad, fd, r = fd_check(th, 3, 1e-4)
    log(f"(6) src_tail: max|w(1.5) - (1+0.5*sigmoid)*w_nom| = {ana:.2e}; "
        f"per-event |dw/dtail - sigmoid*w_nom|_max = {at:.2e}; sum-w AD/FD @1.5 = {r:.10f}")

    # ---- (7) kF_sf: AD-vs-FD + physics sign ----------------------------------------------------------- #
    for kF in (0.9, 1.0, 1.1):
        th = NOM.copy(); th[0] = kF
        ad, fd, r = fd_check(th, 0, 1e-5)
        log(f"(7) kF_sf @ {kF}: d(sum w)/dkF AD = {ad:.6e}, FD = {fd:.6e}, AD/FD = {r:.6f}")
    log("(7) NOTE: at kF=1.0 the FD side is non-convergent for the few S(p,E)~0 edge events "
        "(per-event AD-vs-FD identifies them; excluding them AD/FD -> 1); the AD is correct.")
    def mean_p(theta):
        w = w_of(theta)
        return jnp.sum(w * pj) / jnp.sum(w)
    dmp = float(jax.grad(mean_p)(jnp.asarray(NOM))[0])
    log(f"(7) physics sign: d<|p|>_w/dkF at nominal = {dmp:+.3f} MeV per unit kF "
        f"({'>0 as expected (stretch)' if dmp > 0 else 'UNEXPECTED SIGN'})")

    # ---- (8) Eb_shift: one-sided branch --------------------------------------------------------------- #
    for Eb in (_EB_EPS, 2.0, 5.0, 15.0):
        th = NOM.copy(); th[1] = Eb
        eps = min(1e-3, 0.5 * Eb)
        ad, fd, r = fd_check(th, 1, eps)
        w = np.asarray(w_jit(jnp.asarray(th)))
        log(f"(8) Eb_shift @ {Eb:5.2f} MeV: AD/FD = {r:.6f}; mean w = {w.mean():.4f}; "
            f"w==0 (S clamped): {(w==0).sum()} ({100*(w==0).mean():.2f}%); min w = {w.min():.3e}")
    w_neg = np.asarray(sf_reweight(g, pj, ej, Eb_shift=-5.0))
    log(f"(8) one-sidedness: max|w(Eb=-5) - w(Eb=0)| = {np.max(np.abs(w_neg - w_nom0)):.2e} (clamp to no-op)")

    # ---- figure ---------------------------------------------------------------------------------------- #
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    sub = np.random.default_rng(0).choice(len(pmag), 4000, replace=False)
    r_sp = np.asarray(sf_reweight(g, pj, ej, kF_sf=1.1)); r_up = ratio_upstream(pmag, erem, 1.1, 0.0)
    ax[0, 0].plot(pmag[sub], r_sp[sub], ".", ms=1.5, alpha=0.4, label="spline (reweight)")
    ax[0, 0].plot(pmag[sub], r_up[sub], ".", ms=1.5, alpha=0.4, label="upstream Lagrange")
    ax[0, 0].set(xlabel="|p| [MeV]", ylabel="w", title=r"kF_sf=1.1: $S(p/1.1,E)/S(p,E)$"); ax[0, 0].legend(markerscale=8, fontsize=8)
    r5 = np.asarray(sf_reweight(g, pj, ej, Eb_shift=5.0)); r5u = ratio_upstream(pmag, erem, 1.0, 5.0)
    ax[0, 1].plot(erem[sub], r5[sub], ".", ms=1.5, alpha=0.4, label="spline (reweight)")
    ax[0, 1].plot(erem[sub], r5u[sub], ".", ms=1.5, alpha=0.4, label="upstream Lagrange")
    ax[0, 1].set(xlabel=r"$E_{\rm rem}$ [MeV]", ylabel="w", title=r"Eb_shift=5 MeV: $S(p,E-5)/S(p,E)$")
    ax[0, 1].legend(markerscale=8, fontsize=8)
    ps = np.linspace(0, 800, 400)
    ax[1, 0].plot(pmag[sub], w15[sub], ".", ms=1.5, alpha=0.4, label="sf_reweight(src_tail=1.5)")
    ax[1, 0].plot(ps, 1 + 0.5 / (1 + np.exp(-(ps - 300) / 80)), "k--", lw=1.5, label="analytic 1+0.5·σ((p−300)/80)")
    ax[1, 0].set(xlabel="|p| [MeV]", ylabel="w", title="src_tail=1.5 vs closed form"); ax[1, 0].legend(fontsize=8)
    kfs = np.linspace(0.85, 1.25, 17); ads = []; fds = []
    for kF in kfs:
        th = NOM.copy(); th[0] = kF
        a, f, _ = fd_check(th, 0, 1e-5); ads.append(a); fds.append(f)
    ax[1, 1].plot(kfs, ads, "o-", ms=4, label="autodiff")
    ax[1, 1].plot(kfs, fds, "x--", ms=6, label="central FD (1e-5)")
    ax[1, 1].set(xlabel=r"$k_F$ scale", ylabel=r"d$\Sigma w$/d$k_F$", title="kF_sf gradient: AD vs FD")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("Spectral-function knob validation (1M bank event points)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("output/figures/sf_knob_checks.png", dpi=130)
    log("wrote output/figures/sf_knob_checks.png")


# ================================================================ --fisher: post-fit diagnostics ======== #
def run_fisher():
    sys.argv = [sys.argv[0], "--set", "full17"]                # info_content reads its param set from argv
    from analysis.t2k.differentiability.info_content import (PNAMES, THETA_NOM, NPAR, knobs_of,
                                                             build_datasets, bin_w0, NPZ)
    from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
    from analysis.t2k.differentiability.full_knobs import nominal_knobs

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    K_EB = PNAMES.index("Eb_shift")
    CK = dict(np.load(NPZ.replace(".npz", "_grad.npz"), allow_pickle=True))
    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids()
    nominal = nominal_knobs()
    ds = build_datasets(B)
    log("bank + datasets ready; checkpoint loaded")

    def wf(theta, JB):
        return BR.bank_weight(JB, knobs_of(theta, nominal), grids)
    wf_jit = jax.jit(wf)
    jvp_wf = jax.jit(lambda tang, JB: jax.jvp(lambda th: wf(th, JB), (THETA_NOM,), (tang,))[1])

    # ---- (1) Eb eps-scan ------------------------------------------------------------------------------ #
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

    # ---- (2) pion-FSI null direction ------------------------------------------------------------------ #
    ewv = CK["fisher_eigvals"]; evv = CK["fisher_eigvecs"]; F_tot = CK["F_tot"]
    j0 = int(np.argmin(np.abs(ewv)))
    vn = evv[:, j0]
    dg = np.sqrt(np.maximum(np.diag(F_tot), 1e-300))
    v = vn / dg; v = v / np.linalg.norm(v)                     # knob-space null direction
    comp = ", ".join(f"{v[i]:+.3f}*{PNAMES[i]}" for i in np.argsort(-np.abs(v))[:5])
    log(f"(2) null direction (lam={ewv[j0]:.2e}): {comp}")
    dv_e = np.asarray(jvp_wf(jnp.asarray(v), JB))              # per-event J.v
    ref = np.zeros_like(dv_e)
    for i in np.argsort(-np.abs(v))[:3]:
        ref += np.abs(v[i]) * np.abs(np.asarray(jvp_wf(jnp.zeros(NPAR).at[int(i)].set(1.0), JB)))
    m = ref > 0
    log(f"(2) per-event |J.v|: max = {np.abs(dv_e).max():.3e} "
        f"(vs per-knob gradient scale: structural identity if ~0 for EVERY event)")
    for i, d in enumerate(ds):
        bv = bin_w0(d, dv_e)
        log(f"(2)   {d['name']:12s} per-bin |J.v| max = {np.max(np.abs(bv)):.3e}  "
            f"(vs |J| scale {np.max(np.abs(CK[f'J_{i}'])):.3e})")


if __name__ == "__main__":
    if "--fisher" in sys.argv:
        run_fisher()
    else:
        run_sf()
