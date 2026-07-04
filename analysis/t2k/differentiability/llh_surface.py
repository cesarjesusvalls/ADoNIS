"""Exact multi-parameter LLH (χ², full covariance) surfaces vs the Gaussian/Hessian ellipse at the BFP.

Beyond "FD-Hessian-at-BFP": the differentiable event-bank reweight (`bank_reweight.weight_jit`, 26 knobs, no
Taylor) lets the WHOLE Δχ²(θ) surface be mapped, not just its curvature at one point.  The CC0π-signal mask
and the per-event bin index are θ-independent (frozen kinematics), so a bin content is
`segment_sum(w(θ)·mask, bin_idx)` and χ²(θ) is jit/grad/hessian-able in every knob.  Slicing the bank to the
CC0π-signal events (~340k of 1.87M) makes one exact reweight ~53 ms, so dense 2D/3D grids are cheap.

The Gaussian/Hessian ellipse is EXACT iff the model is linear in θ (a per-channel normalization → residual
1e-12) and fails predictably otherwise: hard-vertex knobs make a curved "banana"; pion-FSI knobs form a
near-flat valley/plane (the flagged equal-rescale invariance of {sabs,s_piN_elastic,s_piN_cex,s_conv}).

CLI modes (run from the repo root, venv active, $ADONIS_EVENT_BANK set):
  python -u analysis/t2k/differentiability/llh_surface.py --gates              # validation gates
  python -u analysis/t2k/differentiability/llh_surface.py --2d [pairs] [--ng 61] [--obs dpt,dat]
  python -u analysis/t2k/differentiability/llh_surface.py --3d [triples] [--ng 25]
  python -u analysis/t2k/differentiability/llh_surface.py --combo [pairs] [--ng 51]   # CC0π+CC1π + overlay
  python -u analysis/t2k/differentiability/llh_surface.py --plot-only <npz>    # re-render (2D or 3D auto)
Pairs:   maqe_axial | sabs_selastic | qe_res_norm      Triples: ma_axial_resnorm | fsi_trio
Figures -> output/figures/llh_*   npz -> /tmp/adonis_tune_runs/llh_*   Evidence: docs/logbook/llh_surface.md
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))       # repo root
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.optimize as sopt

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability import info_content as IC       # load_cc0pi + build_datasets (SSOT)
from analysis.t2k.differentiability.full_knobs import nominal_knobs

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
OUTDIR = "/tmp/adonis_tune_runs"                                    # run histories/caches (README convention)
CLIP = (0.3, 3.0)                                                  # global knob clip rails (physical)


# ================================================================ core: robust χ² over the sliced bank ===== #
def robust_cinv(cov, rtol=1e-10):
    """Eigen-floored inverse of a symmetric covariance (SVD/pinv with a relative cutoff `rtol·max(eig)`) so an
    ill-conditioned block cannot blow up χ² (the 2D-pcos covariance needed this).  Returns (Cinv, cond, nfloored)."""
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    wmax = float(w.max()); floor = rtol * wmax
    nfloored = int(np.sum(w < floor))
    cinv = (V * (1.0 / np.where(w < floor, floor, w))) @ V.T
    cond = wmax / float(w.min()) if w.min() > 0 else np.inf
    return 0.5 * (cinv + cinv.T), cond, nfloored


def load_signal_bank(bankdir=None, verbose=True):
    """Load the 1M bank, slice to CC0π-signal events, return (JBs, grids, aux).  JBs is the on-device pytree
    for weight_jit over signal events only; aux carries the leading proton + full B (for observables)."""
    bankdir = bankdir or BANKDIR
    B = BP.load_bank(bankdir)
    mask, lead = BP.signal_cc0pi(B)                                 # prim-pi absorbed + T2K acceptance
    mask = np.asarray(mask); nsig = int(mask.sum())
    if verbose:
        print(f"[llh] bank {bankdir}: {len(B['w0'])} events -> {nsig} CC0pi signal "
              f"({(B['channel'][mask] == 0).sum()} QE + {(B['channel'][mask] == 1).sum()} RES)", flush=True)
    JB = BR.to_jax(B)
    JBs = {k: v[jnp.asarray(mask)] for k, v in JB.items()}
    return JBs, BR.default_grids(), dict(B=B, mask=mask, lead=lead, nsig=nsig)


def obs_binning(aux, obs):
    """Per-signal-event bin index + T2K CC0π STV data spec for an observable (reuses info_content.load_cc0pi)."""
    B, mask, lead = aux["B"], aux["mask"], aux["lead"]
    edges, conv, data, cov = IC.load_cc0pi(obs)
    val = np.asarray(BP.dpt(B, lead) if obs == "dpt" else BP.dat(B, lead))[mask]
    nb = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, val) - 1, 0, nb - 1)
    return dict(obs=obs, idx=jnp.asarray(idx), nb=nb, bw=jnp.asarray(np.diff(edges)), conv=conv,
                edges=edges, data=jnp.asarray(data), cov=cov)


def make_chi2(JBs, grids, obs_specs, rtol=1e-10):
    """Build model_vec(knobs), chi2_abs, chi2_shape (global-A profiled), chi2_both over a LIST of obs specs.
    ONE exact weight per call, binned per observable; block-diagonal covariance across observables.

    CONSTANT-FOLDING NaN: the weight MUST be `BR.weight_jit(JBs, knobs, grids)` with JBs/grids as jit OPERANDS.
    A jit that closes over JBs constant-folds the 21.8M-row FSI gather to (value-dependent) NaN, so model_vec/
    chi2 are deliberately NOT wrapped in jax.jit (that recaptures JBs); speed comes from weight_jit, and
    jax.grad/jax.hessian trace THROUGH it with JBs still an operand."""
    idxs = [s["idx"] for s in obs_specs]; nbs = [s["nb"] for s in obs_specs]
    bws = [s["bw"] for s in obs_specs]; convs = [s["conv"] for s in obs_specs]
    cinvs, conds, nfl = [], [], []
    for s in obs_specs:
        ci, cond, nf = robust_cinv(s["cov"], rtol=rtol); cinvs.append(ci); conds.append(cond); nfl.append(nf)
    data_all = jnp.asarray(np.concatenate([np.asarray(s["data"]) for s in obs_specs]))
    import scipy.linalg as sla
    cinv_all = jnp.asarray(sla.block_diag(*cinvs))

    def model_vec(knobs):
        w = BR.weight_jit(JBs, knobs, grids)                       # JBs/grids OPERANDS -> finite
        return jnp.concatenate([jax.ops.segment_sum(w, idx, num_segments=nb) / bw * conv
                                for idx, nb, bw, conv in zip(idxs, nbs, bws, convs)])

    def chi2_abs(knobs):
        r = model_vec(knobs) - data_all
        return r @ cinv_all @ r

    def chi2_shape(knobs):
        m = model_vec(knobs); A = (m @ cinv_all @ data_all) / (m @ cinv_all @ m)
        r = A * m - data_all
        return r @ cinv_all @ r

    def chi2_both(knobs):
        m = model_vec(knobs); ra = m - data_all
        A = (m @ cinv_all @ data_all) / (m @ cinv_all @ m); rs = A * m - data_all
        return ra @ cinv_all @ ra, rs @ cinv_all @ rs, A

    return dict(model_vec=model_vec, chi2_abs=chi2_abs, chi2_shape=chi2_shape, chi2_both=chi2_both,
                data_all=data_all, cinv_all=cinv_all, conds=conds, nfloored=nfl,
                ndf=int(len(data_all)), obs_names=[s["obs"] for s in obs_specs])


def assembler(names):
    """Return assemble(p) that overrides the named SCALAR knobs in nominal_knobs by the vector p."""
    NOM = nominal_knobs()
    for n in names:
        assert not isinstance(NOM[n], tuple), f"{n} is a tuple knob; scalar knobs only here"

    def assemble(p):
        k = dict(NOM)
        for i, n in enumerate(names):
            k[n] = p[i]
        return k
    return assemble, NOM


def _fit_bfp(chi2_pair, p_start, bounds):
    """Box-constrained BFP via L-BFGS-B with the exact jax gradient (chi2_pair calls weight_jit -> JBs stays
    an operand; grad is autodiff)."""
    vg = jax.value_and_grad(chi2_pair)
    def f(p):
        v, g = vg(jnp.asarray(p)); return float(v), np.asarray(g, float)
    r = sopt.minimize(f, np.asarray(p_start, float), jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(maxiter=400, ftol=1e-12, gtol=1e-10))
    return r.x, float(r.fun)


# ================================================================ 2D surfaces ============================== #
# pair key -> (name0, name1, tex0, tex1, (lo0,hi0), (lo1,hi1))
PAIRS = {
    "maqe_axial":    ("M_A_qe", "axial_strength", r"$M_A^{\rm QE}$", r"$g_A^{\rm QE}$",
                      (0.62, 1.70), (0.48, 1.45)),
    "sabs_selastic": ("sabs", "s_piN_elastic", r"$\sigma_{\pi\,\rm abs}$", r"$\sigma_{\pi N\,\rm el}$",
                      (0.35, 2.30), (0.30, 2.70)),
    "qe_res_norm":   ("qe_norm", "res_norm", r"QE norm", r"RES norm",
                      (0.78, 1.14), (0.30, 2.80)),
}


def _nongauss(dchi_true, dchi_gauss):
    """Non-Gaussianity metrics on the region Δχ²_true < 9 (3σ)."""
    reg = dchi_true < 9.0; resid = dchi_true - dchi_gauss
    out = dict(max_absdev=float(np.max(np.abs(resid[reg]))), rms_dev=float(np.sqrt(np.mean(resid[reg] ** 2))),
               frac_max=float(np.max(np.abs(resid[reg])) / 9.0),
               area_true_1s=float(np.sum(dchi_true < 1.0)), area_gauss_1s=float(np.sum(dchi_gauss < 1.0)))
    out["area_ratio_1s"] = out["area_true_1s"] / max(out["area_gauss_1s"], 1.0)
    return out


def compute_pair(C, key, ng, obs_tag):
    """Map the exact Δχ² (abs) over a dense 2D grid for pair `key`; fit the BFP + exact Hessian; quantify
    non-Gaussianity; persist an npz.  C: a make_chi2 / combined dict (chi2_abs, chi2_both, ndf)."""
    name0, name1, tex0, tex1, r0, r1 = PAIRS[key]
    assemble, NOM = assembler([name0, name1])
    chi2_abs_pair = lambda p: C["chi2_abs"](assemble(p))          # NOT jitted (JBs must stay an operand)
    p_nom = np.array([float(NOM[name0]), float(NOM[name1])])
    bounds = [(max(CLIP[0], r0[0] - 0.2), min(CLIP[1], r0[1] + 0.2)),
              (max(CLIP[0], r1[0] - 0.2), min(CLIP[1], r1[1] + 0.2))]
    t0 = time.time()
    def log(m): print(f"  [{key}|{obs_tag}] [{time.time()-t0:5.1f}s] {m}", flush=True)

    bfp, chi2_min = _fit_bfp(chi2_abs_pair, p_nom, bounds)
    H = np.asarray(jax.hessian(chi2_abs_pair)(jnp.asarray(bfp)))
    try:
        V = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        V = 2.0 * np.linalg.pinv(H)
    rail = [(abs(bfp[i] - bounds[i][0]) < 1e-6) or (abs(bfp[i] - bounds[i][1]) < 1e-6) for i in range(2)]
    log(f"BFP=({bfp[0]:.4f},{bfp[1]:.4f})  chi2/ndf={chi2_min/C['ndf']:.3f}  rail={rail}  V=2H^-1={V.tolist()}")

    g0 = np.linspace(r0[0], r0[1], ng); g1 = np.linspace(r1[0], r1[1], ng)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    chi2_grid = np.zeros((ng, ng)); chi2s_grid = np.zeros((ng, ng)); A_grid = np.zeros((ng, ng))
    both = C["chi2_both"]; tstart = time.time()
    for i in range(ng):
        for j in range(ng):
            k = dict(NOM); k[name0] = float(G0[i, j]); k[name1] = float(G1[i, j])
            ca, cs, A = both(k); chi2_grid[i, j] = float(ca); chi2s_grid[i, j] = float(cs); A_grid[i, j] = float(A)
        if i % max(1, ng // 6) == 0:
            log(f"grid row {i+1}/{ng}  ({(time.time()-tstart)/((i+1)*ng)*1e3:.0f} ms/eval)")
    dchi_true = chi2_grid - chi2_min
    dchi_shape = chi2s_grid - float(chi2s_grid.min())
    dP0 = G0 - bfp[0]; dP1 = G1 - bfp[1]; Hh = 0.5 * H
    dchi_gauss = Hh[0, 0] * dP0 ** 2 + 2 * Hh[0, 1] * dP0 * dP1 + Hh[1, 1] * dP1 ** 2
    ng_m = _nongauss(dchi_true, dchi_gauss)
    log(f"non-Gaussianity: max|dev|={ng_m['max_absdev']:.2f} (frac {ng_m['frac_max']:.2f})  "
        f"RMS={ng_m['rms_dev']:.2f}  area(true/gauss @1s)={ng_m['area_ratio_1s']:.2f}")

    os.makedirs(OUTDIR, exist_ok=True); npz = f"{OUTDIR}/llh_{key}_{obs_tag}.npz"
    np.savez(npz, kind="2d", key=key, obs_tag=obs_tag, name0=name0, name1=name1, tex0=tex0, tex1=tex1,
             g0=g0, g1=g1, chi2_grid=chi2_grid, chi2s_grid=chi2s_grid, A_grid=A_grid,
             dchi_true=dchi_true, dchi_shape=dchi_shape, dchi_gauss=dchi_gauss,
             bfp=bfp, chi2_min=chi2_min, H=H, V=V, rail=np.array(rail), p_nom=p_nom, ndf=C["ndf"],
             bounds=np.array(bounds), **{f"ng_{kk}": vv for kk, vv in ng_m.items()})
    log(f"saved -> {npz}"); return npz


def make_figure_2d(npz):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz, allow_pickle=True)
    g0, g1 = d["g0"], d["g1"]; G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    dchi_true, dchi_gauss = d["dchi_true"], d["dchi_gauss"]
    bfp, pnom = d["bfp"], d["p_nom"]; tex0, tex1 = str(d["tex0"]), str(d["tex1"])
    key, obs_tag = str(d["key"]), str(d["obs_tag"]); levels = [1.0, 4.0, 9.0]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    a = ax[0]
    pc = a.pcolormesh(G0, G1, np.clip(dchi_true, 0, 25), cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=a, fraction=0.046, label=r"$\Delta\chi^2$ (true, capped 25)")
    cs_t = a.contour(G0, G1, dchi_true, levels=levels, colors="w", linewidths=1.8)
    a.clabel(cs_t, fmt={1.0: "1", 4.0: "4", 9.0: "9"}, fontsize=8)
    a.contour(G0, G1, dchi_gauss, levels=levels, colors="r", linewidths=1.4, linestyles="--")
    a.plot(pnom[0], pnom[1], "s", color="0.8", ms=9, mec="k", label="nominal")
    a.plot(bfp[0], bfp[1], "X", color="red", ms=12, mec="k", label="BFP")
    a.plot([], [], "w-", label=r"true $\Delta\chi^2$=1,4,9"); a.plot([], [], "r--", label="Hessian ellipse")
    a.set_xlabel(tex0); a.set_ylabel(tex1); a.legend(fontsize=8, loc="best")
    a.set_title(f"True LLH vs Hessian ellipse  ({key})")
    a = ax[1]
    prof0 = dchi_true.min(axis=1); cond0 = dchi_true[:, np.argmin(np.abs(g1 - bfp[1]))]
    a.plot(g0, prof0, "C0-", lw=2, label=r"profiled $\Delta\chi^2(p_0)=\min_{p_1}$")
    a.plot(g0, cond0, "C0--", lw=1.4, label=r"conditional (slice at BFP $p_1$)")
    sig0 = np.sqrt(abs(d["V"][0, 0]))
    a.plot(g0, ((g0 - bfp[0]) / sig0) ** 2, "r:", lw=1.4, label=r"Gaussian ($\sigma$ from $V_{00}$)")
    a.axhline(1, color="0.6", lw=0.7, ls=":"); a.set_ylim(0, 12)
    a.set_xlabel(tex0); a.set_ylabel(r"$\Delta\chi^2$"); a.legend(fontsize=8)
    a.set_title("Profiled vs conditional vs Gaussian (along $p_0$)")
    a = ax[2]; resid = dchi_true - dchi_gauss
    vmax = float(np.percentile(np.abs(resid[dchi_true < 12]), 99)) or 1.0
    pc = a.pcolormesh(G0, G1, resid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(pc, ax=a, fraction=0.046, label=r"$\Delta\chi^2_{\rm true}-\Delta\chi^2_{\rm Gauss}$")
    a.contour(G0, G1, dchi_true, levels=[1, 4, 9], colors="k", linewidths=0.8, alpha=0.5)
    a.plot(bfp[0], bfp[1], "X", color="k", ms=10); a.set_xlabel(tex0); a.set_ylabel(tex1)
    a.set_title(f"non-Gaussianity  max|dev|={float(d['ng_max_absdev']):.1f}  "
                f"area ratio@1$\\sigma$={float(d['ng_area_ratio_1s']):.2f}")
    rail = d["rail"]; chan = "CC0$\\pi$+CC1$\\pi$" if obs_tag.startswith("cc0cc1") else "CC0$\\pi$"
    fig.suptitle(f"T2K {chan} [{obs_tag}] exact LLH surface — {tex0} $\\times$ {tex1}   "
                 f"(BFP $\\chi^2$/ndf={float(d['chi2_min'])/int(d['ndf']):.2f}"
                 f"{'  RAIL!' if bool(rail[0]) or bool(rail[1]) else ''})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); os.makedirs("output/figures", exist_ok=True)
    out = f"output/figures/llh_surface_{key}_{obs_tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig); print(f"wrote {out}", flush=True)


# ================================================================ 3D slices ================================ #
TRIPLES = {
    # M_A_qe x axial x res_norm: the QE-axial banana in (M_A_qe, axial) + a soft res_norm axis -> interior BFP.
    "ma_axial_resnorm": (["M_A_qe", "axial_strength", "res_norm"],
                         [r"$M_A^{\rm QE}$", r"$g_A^{\rm QE}$", "RES norm"],
                         [(0.65, 1.70), (0.50, 1.45), (0.30, 2.70)]),
    # sabs x s_piN_elastic x s_piN_cex: 3 of the 4 knobs in the flagged EXACT pion-FSI flat direction (equal
    # rescale of {sabs,s_piN_elastic,s_piN_cex,s_conv} is a per-event invariance) -> near-singular Hessian.
    "fsi_trio": (["sabs", "s_piN_elastic", "s_piN_cex"],
                 [r"$\sigma_{\pi\,\rm abs}$", r"$\sigma_{\pi N\,\rm el}$", r"$\sigma_{\pi N\,\rm cex}$"],
                 [(0.40, 2.20), (0.35, 2.60), (0.35, 2.60)]),
}


def compute_3d(C, key, ng):
    names, texs, rngs = TRIPLES[key]
    assemble, NOM = assembler(names)
    chi2 = lambda p: C["chi2_abs"](assemble(p))
    p_nom = np.array([float(NOM[n]) for n in names])
    bounds = [(max(CLIP[0], r[0] - 0.2), min(CLIP[1], r[1] + 0.2)) for r in rngs]
    t0 = time.time()
    def log(m): print(f"  [{key}] [{time.time()-t0:5.1f}s] {m}", flush=True)
    bfp, chi2_min = _fit_bfp(chi2, p_nom, bounds)
    H = np.asarray(jax.hessian(chi2)(jnp.asarray(bfp)))
    rail = [(abs(bfp[i] - bounds[i][0]) < 1e-6) or (abs(bfp[i] - bounds[i][1]) < 1e-6) for i in range(3)]
    dg = np.sqrt(np.maximum(np.diag(H), 1e-300)); Hn = H / np.outer(dg, dg)
    ew, ev = np.linalg.eigh(Hn)
    log(f"BFP={np.round(bfp,4)}  chi2/ndf={chi2_min/C['ndf']:.3f}  rail={rail}  cond(H)={np.linalg.cond(H):.3e}")
    for j in range(3):
        comp = " ".join(f"{ev[t,j]:+.2f}*{names[t]}" for t in np.argsort(-np.abs(ev[:, j])))
        log(f"    lam={ew[j]:.3e}  dir: {comp}")
    axes = [np.linspace(r[0], r[1], ng) for r in rngs]; cube = np.zeros((ng, ng, ng))
    both = C["chi2_abs"]; tstart = time.time()
    for i in range(ng):
        ki = float(axes[0][i])
        for j in range(ng):
            kj = float(axes[1][j])
            for l in range(ng):
                k = dict(NOM); k[names[0]] = ki; k[names[1]] = kj; k[names[2]] = float(axes[2][l])
                cube[i, j, l] = float(both(k))
        if i % max(1, ng // 8) == 0:
            log(f"cube slab {i+1}/{ng}  ({(time.time()-tstart)/((i+1)*ng*ng)*1e3:.0f} ms/eval)")
    os.makedirs(OUTDIR, exist_ok=True); npz = f"{OUTDIR}/llh_{key}_3d.npz"
    np.savez(npz, kind="3d", key=key, names=np.array(names), texs=np.array(texs),
             a0=axes[0], a1=axes[1], a2=axes[2], dcube=cube - chi2_min, cube=cube,
             bfp=bfp, chi2_min=chi2_min, H=H, Hn=Hn, eigval=ew, eigvec=ev,
             rail=np.array(rail), p_nom=p_nom, ndf=C["ndf"], cond=np.linalg.cond(H))
    log(f"saved -> {npz}"); return npz


def make_figure_3d(npz):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz, allow_pickle=True)
    a = [d["a0"], d["a1"], d["a2"]]; texs = [str(x) for x in d["texs"]]
    dcube = d["dcube"]; bfp = d["bfp"]; H = d["H"]; ew = d["eigval"]; ev = d["eigvec"]
    names = [str(x) for x in d["names"]]; key = str(d["key"]); pairs = [(0, 1), (0, 2), (1, 2)]
    levels = [1.0, 4.0, 9.0]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    for pi, (u, v) in enumerate(pairs):
        rem = ({0, 1, 2} - {u, v}).pop(); P = dcube.min(axis=rem)      # profiled 2D (min over the 3rd axis)
        U, Vv = np.meshgrid(a[u], a[v], indexing="ij"); aa = ax[pi]
        pc = aa.pcolormesh(U, Vv, np.clip(P, 0, 20), cmap="viridis", shading="auto")
        cs = aa.contour(U, Vv, P, levels=levels, colors="w", linewidths=1.6)
        aa.clabel(cs, fmt={1: "1", 4: "4", 9: "9"}, fontsize=7)
        try:                                                          # projected (profiled) Hessian ellipse
            Vfull = 2.0 * np.linalg.inv(H); Vp = np.array([[Vfull[u, u], Vfull[u, v]], [Vfull[v, u], Vfull[v, v]]])
            wv, vv = np.linalg.eigh(np.linalg.inv(Vp)); th = np.linspace(0, 2 * np.pi, 120)
            for Ln in (1.0, 4.0):
                xy = vv @ (np.sqrt(Ln / np.maximum(wv, 1e-12))[:, None] * np.array([np.cos(th), np.sin(th)]))
                aa.plot(bfp[u] + xy[0], bfp[v] + xy[1], "r--", lw=1.1)
        except np.linalg.LinAlgError:
            pass
        aa.plot(bfp[u], bfp[v], "X", color="red", ms=11, mec="k")
        aa.set_xlim(a[u].min(), a[u].max()); aa.set_ylim(a[v].min(), a[v].max())   # clip runaway flat ellipse
        aa.set_xlabel(texs[u]); aa.set_ylabel(texs[v])
        aa.set_title(f"profiled $\\Delta\\chi^2$ (min over {texs[rem]})", fontsize=9)
        fig.colorbar(pc, ax=aa, fraction=0.046)
    aa = ax[3]; order = np.argsort(ew)
    aa.bar(range(3), ew[order], color=["C3", "C1", "C0"])
    for r, j in enumerate(order):
        lab = "\n".join(f"{ev[t,j]:+.2f} {names[t]}" for t in np.argsort(-np.abs(ev[:, j])))
        aa.text(r, ew[j] + 0.02 * ew.max(), lab, ha="center", va="bottom", fontsize=6.5)
    aa.set_yscale("log"); aa.set_xticks(range(3)); aa.set_xticklabels(["flat", "mid", "stiff"])
    aa.set_ylabel("normalized-Hessian eigenvalue"); aa.set_title(f"eigen-spectrum  cond(H)={float(d['cond']):.1e}", fontsize=9)
    rail = d["rail"]
    fig.suptitle(f"T2K CC0$\\pi$ [dpt+dat] 3D LLH slice — {texs[0]} $\\times$ {texs[1]} $\\times$ {texs[2]}  "
                 f"(BFP $\\chi^2$/ndf={float(d['chi2_min'])/int(d['ndf']):.2f}"
                 f"{'  RAIL!' if any(bool(x) for x in rail) else ''})  — red dashed = profiled Hessian ellipse",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs("output/figures", exist_ok=True)
    out = f"output/figures/llh_3d_{key}.png"
    fig.savefig(out, dpi=130); plt.close(fig); print(f"wrote {out}", flush=True)


# ================================================================ CC0π+CC1π combination ==================== #
# CC1pi tightens RES -> zoom the combined norm-pair grid onto the now-narrow constraint (the CC0pi-only range
# res_norm (0.30,2.80) is far too wide once CC1pi is added).  maqe_axial is QE-only (CC1pi orthogonal) -> kept.
COMB_RANGES = {"qe_res_norm": ((0.76, 1.02), (0.45, 1.45))}


def make_chi2_combined(verbose=True):
    """make_chi2-compatible dict for the 5-dataset CC0π+CC1π combined absolute χ², using a UNION-sliced bank.
    Dataset assembly (acceptance, units, FROZEN nominal free-H offset) is reused from info_content.build_datasets."""
    B = BP.load_bank(BANKDIR); ds = IC.build_datasets(B); N = len(B["w0"])
    allsel = np.unique(np.concatenate([np.asarray(d["sel_idx"]) for d in ds])).astype(np.int64)
    pos = -np.ones(N, np.int64); pos[allsel] = np.arange(len(allsel))          # full -> union-slice position
    JBu = {k: v[jnp.asarray(allsel)] for k, v in BR.to_jax(B).items()}
    grids = BR.default_grids()
    if verbose:
        print(f"[comb] bank {N} ev -> union slice {len(allsel)} "
              f"(CC0pi {int(BP.signal_cc0pi(B)[0].sum())} + CC1pi {int(BP.signal_cc1pi_stv(B)[0].sum())})", flush=True)
    STR, datas, cinvs, conds = [], [], [], []
    for d in ds:
        sel_u = pos[np.asarray(d["sel_idx"])]; assert (sel_u >= 0).all()
        STR.append(dict(sel=jnp.asarray(sel_u), bidx=jnp.asarray(np.asarray(d["binidx"])), nb=int(d["nbin"]),
                        scale=jnp.asarray(np.asarray(d["scale_bin"])), off=jnp.asarray(np.asarray(d["offset"]))))
        datas.append(np.asarray(d["data"])); cinvs.append(np.asarray(d["Cinv"])); conds.append(float(np.linalg.cond(d["Cinv"])))
    data_all = jnp.asarray(np.concatenate(datas))
    import scipy.linalg as sla
    cinv_all = jnp.asarray(sla.block_diag(*cinvs)); ndf = int(len(data_all)); names = [d["name"] for d in ds]

    def model_vec(knobs):
        w = BR.weight_jit(JBu, knobs, grids)                       # JBu operand -> finite
        return jnp.concatenate([jax.ops.segment_sum(w[s["sel"]], s["bidx"], num_segments=s["nb"]) * s["scale"] + s["off"]
                                for s in STR])

    def chi2_abs(knobs):
        r = model_vec(knobs) - data_all
        return r @ cinv_all @ r

    def chi2_both(knobs):                                          # single global A across mixed units (secondary)
        m = model_vec(knobs); ra = m - data_all
        A = (m @ cinv_all @ data_all) / (m @ cinv_all @ m); rs = A * m - data_all
        return ra @ cinv_all @ ra, rs @ cinv_all @ rs, A

    if verbose:
        c0 = float(chi2_abs(nominal_knobs()))
        print(f"[comb] nominal combined chi2/ndf = {c0/ndf:.3f} (ndf={ndf})", flush=True)
        for nm, cd in zip(names, conds):
            print(f"[comb] cov[{nm}] cond={cd:.2e}", flush=True)
    return dict(model_vec=model_vec, chi2_abs=chi2_abs, chi2_both=chi2_both, ndf=ndf,
                obs_names=names, conds=conds, nfloored=[0] * len(ds))


def make_compare():
    """Overlay the exact qe_norm x res_norm 1σ/2σ ellipses from CC0π-only vs CC0π+CC1π (both surfaces are
    exactly quadratic, so the Hessian ellipse IS the exact contour) -> the degeneracy-breaking figure."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        d0 = np.load(f"{OUTDIR}/llh_qe_res_norm_dptdat.npz", allow_pickle=True)
        dc = np.load(f"{OUTDIR}/llh_qe_res_norm_cc0cc1.npz", allow_pickle=True)
    except FileNotFoundError as e:
        print(f"[compare] need both qe_res_norm npz (run --2d qe_res_norm and --combo qe_res_norm): {e}", flush=True)
        return
    def ell(ax, bfp, V, color, label):
        th = np.linspace(0, 2 * np.pi, 200); w, Vv = np.linalg.eigh(V)
        for kk, ls in ((1.0, "-"), (2.0, "--")):
            xy = Vv @ (np.sqrt(np.maximum(w, 0) * kk)[:, None] * np.array([np.cos(th), np.sin(th)]))
            ax.plot(bfp[0] + xy[0], bfp[1] + xy[1], color=color, ls=ls, lw=2.0,
                    label=f"{label} {int(kk)}$\\sigma$" if kk == 1 else None)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ell(ax, d0["bfp"], d0["V"], "C0", "CC0$\\pi$ (dpt+dat)"); ell(ax, dc["bfp"], dc["V"], "C3", "CC0$\\pi$+CC1$\\pi$")
    ax.plot(*d0["bfp"], "o", color="C0", ms=7, mec="k"); ax.plot(*dc["bfp"], "X", color="C3", ms=11, mec="k")
    ax.plot(1, 1, "s", color="0.5", ms=8, mec="k", label="nominal")
    s0 = np.sqrt(np.diag(d0["V"])); sc = np.sqrt(np.diag(dc["V"]))
    ax.set_xlabel("QE norm"); ax.set_ylabel("RES norm"); ax.set_xlim(0.74, 1.16); ax.grid(alpha=0.3)
    ax.set_title("Adding CC1$\\pi^+$Np breaks the CC0$\\pi$ RES-norm degeneracy\n"
                 f"$\\sigma_{{\\rm RES}}$: {s0[1]:.2f} (CC0$\\pi$) $\\to$ {sc[1]:.2f} (CC0$\\pi$+CC1$\\pi$);   "
                 f"$\\sigma_{{\\rm QE}}$: {s0[0]:.3f} $\\to$ {sc[0]:.3f}", fontsize=10)
    ax.legend(fontsize=9, loc="upper right"); fig.tight_layout()
    out = "output/figures/llh_cc0cc1_norm_degeneracy.png"; os.makedirs("output/figures", exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig); print(f"wrote {out}", flush=True)
    print(f"[compare] CC0pi-only : qe={d0['bfp'][0]:.3f}+/-{s0[0]:.3f}  res={d0['bfp'][1]:.3f}+/-{s0[1]:.3f}", flush=True)
    print(f"[compare] CC0pi+CC1pi: qe={dc['bfp'][0]:.3f}+/-{sc[0]:.3f}  res={dc['bfp'][1]:.3f}+/-{sc[1]:.3f}", flush=True)


# ================================================================ gates ==================================== #
def run_gates():
    """Validation gates: (A) slicing bit-exactness, (B) nominal footprint vs the bare-w0 forward (informational),
    (C) AD-Hessian vs central FD, + covariance conditioning and nominal joint χ²/ndf."""
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    JBs, grids, aux = load_signal_bank(); NOM = nominal_knobs()
    B, mask = aux["B"], aux["mask"]
    wfull = np.asarray(BR.weight_jit(BR.to_jax(B), NOM, grids)); wslice = np.asarray(BR.weight_jit(JBs, NOM, grids))
    relA = abs(float(wfull[mask].sum()) - float(wslice.sum())) / abs(float(wfull[mask].sum()))
    log(f"GATE A slicing: rel={relA:.2e}  {'PASS' if relA < 1e-12 else 'FAIL'}")
    specs = [obs_binning(aux, o) for o in ("dpt", "dat")]; C = make_chi2(JBs, grids, specs)
    mvec = np.asarray(C["model_vec"](NOM)); lead = aux["lead"]; w0 = B["w0"].astype(np.float64); off = 0
    for s in specs:
        val = np.asarray(BP.dpt(B, lead) if s["obs"] == "dpt" else BP.dat(B, lead))
        h, _ = np.histogram(val[mask], bins=s["edges"], weights=w0[mask]); fwd = h / np.diff(s["edges"]) * s["conv"]
        mv = mvec[off:off + s["nb"]]; off += s["nb"]
        rel = np.max(np.abs(mv - fwd) / np.clip(np.abs(fwd), 1e-30, None)); aggr = abs(mv.sum() - fwd.sum()) / abs(fwd.sum())
        log(f"GATE B footprint[{s['obs']}]: max per-bin={rel:.2e}  aggregate={aggr:.2e}  (nominal SF/FSI reweight; INFORMATIONAL)")
    for nm, cond, nf in zip(C["obs_names"], C["conds"], C["nfloored"]):
        log(f"cov[{nm}]: cond={cond:.3e}  eigs_floored={nf}")
    log(f"nominal chi2/ndf: abs={float(C['chi2_abs'](NOM))/C['ndf']:.3f}  shape={float(C['chi2_shape'](NOM))/C['ndf']:.3f}  (ndf={C['ndf']})")
    assemble, _ = assembler(["M_A_qe", "axial_strength"])
    chi2_pair = lambda p: C["chi2_abs"](assemble(p))              # NOT jitted (JBs operand)
    p0 = jnp.array([1.05, 0.95]); Had = np.asarray(jax.hessian(chi2_pair)(p0)); eps = 1e-4
    f = lambda p: float(chi2_pair(jnp.asarray(p))); p0n = np.asarray(p0); Hfd = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2); ei[i] = eps; ej = np.zeros(2); ej[j] = eps
            Hfd[i, j] = (f(p0n+ei+ej) - f(p0n+ei-ej) - f(p0n-ei+ej) + f(p0n-ei-ej)) / (4*eps*eps)
    relC = np.max(np.abs(Had - Hfd) / np.clip(np.abs(Hfd), 1e-6, None))
    log(f"GATE C AD-Hessian vs FD: relmax={relC:.2e}  {'PASS' if relC < 1e-3 else 'FAIL'}")


# ================================================================ CLI ====================================== #
def _plot_only(npz):
    kind = str(np.load(npz, allow_pickle=True).get("kind", "2d"))
    (make_figure_3d if kind == "3d" else make_figure_2d)(npz)


def main():
    argv = sys.argv[1:]
    def opt(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default
    ng = opt("--ng", None); obs = opt("--obs", "dpt,dat")
    keys = [a for a in argv if not a.startswith("--") and argv[argv.index(a) - 1] not in ("--ng", "--obs")]

    if "--plot-only" in argv:
        _plot_only(opt("--plot-only", None)); return
    if "--gates" in argv:
        run_gates(); return
    t0 = time.time()
    if "--3d" in argv:
        ng = int(ng or 25); ks = [k for k in keys if k in TRIPLES] or list(TRIPLES)
        print(f"[llh --3d] triples={ks} ng={ng}", flush=True)
        JBs, grids, aux = load_signal_bank(); C = make_chi2(JBs, grids, [obs_binning(aux, o) for o in ("dpt", "dat")])
        for k in ks:
            make_figure_3d(compute_3d(C, k, ng))
    elif "--combo" in argv:
        ng = int(ng or 51); ks = [k for k in keys if k in PAIRS] or ["qe_res_norm", "maqe_axial"]
        for k, (r0, r1) in COMB_RANGES.items():
            n0, n1, t0_, t1_, _, _ = PAIRS[k]; PAIRS[k] = (n0, n1, t0_, t1_, r0, r1)   # zoom combined grid
        print(f"[llh --combo] pairs={ks} ng={ng}", flush=True)
        C = make_chi2_combined()
        for k in ks:
            make_figure_2d(compute_pair(C, k, ng, "cc0cc1"))
        make_compare()
    else:   # default: --2d
        ng = int(ng or 61); ks = [k for k in keys if k in PAIRS] or list(PAIRS)
        obs_list = obs.split(","); obs_tag = "".join(obs_list)
        print(f"[llh --2d] pairs={ks} obs={obs_list} ng={ng}", flush=True)
        JBs, grids, aux = load_signal_bank(); C = make_chi2(JBs, grids, [obs_binning(aux, o) for o in obs_list])
        for nm, cond, nf in zip(C["obs_names"], C["conds"], C["nfloored"]):
            print(f"[llh] cov[{nm}] cond={cond:.2e} floored={nf}", flush=True)
        for k in ks:
            make_figure_2d(compute_pair(C, k, ng, obs_tag))
    print(f"[llh] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
