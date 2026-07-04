"""Map the EXACT T2K CC0pi LLH (chi^2, full covariance) over dense 2D parameter grids, and contrast the true
Delta chi^2 contours against the Gaussian ellipse from jax.hessian at the best-fit point (BFP).

Beyond "FD-Hessian-at-BFP": we evaluate the exact chi^2(theta) on a dense grid (one differentiable bank
reweight per grid point, ~73 ms over the 340k CC0pi-signal events), fit the BFP, take the exact Hessian
there, and quantify where the true surface departs from the Gaussian approximation (curved degeneracy
valleys / "bananas", asymmetric tails, rail effects).

  python -u scripts/cc0pi_llh_surface.py            [pair1 pair2 ...]  [--obs dpt,dat] [--ng 61]
  python -u scripts/cc0pi_llh_surface.py --plot-only /tmp/adonis_llh/<pair>_<obs>.npz

Pairs (keys): maqe_axial | sabs_selastic | qe_res_norm   (default: all three).
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/ (cc0pi_llh_lib)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.optimize as sopt

import cc0pi_llh_lib as L

OUTDIR = "/tmp/adonis_llh"
CLIP = (0.3, 3.0)                              # global knob clip rails (physical, matches info_content)

# pair registry: key -> (name0, name1, tex0, tex1, (lo0,hi0), (lo1,hi1))
PAIRS = {
    "maqe_axial":    ("M_A_qe", "axial_strength", r"$M_A^{\rm QE}$", r"$g_A^{\rm QE}$",
                      (0.62, 1.70), (0.48, 1.45)),
    "sabs_selastic": ("sabs", "s_piN_elastic", r"$\sigma_{\pi\,\rm abs}$", r"$\sigma_{\pi N\,\rm el}$",
                      (0.35, 2.30), (0.30, 2.70)),
    "qe_res_norm":   ("qe_norm", "res_norm", r"QE norm", r"RES norm",
                      (0.78, 1.14), (0.30, 2.80)),
}


def _fit_bfp(chi2_pair, p_start, bounds):
    """Box-constrained BFP via L-BFGS-B with the exact jax gradient (chi2_pair calls weight_jit -> JBs stays
    an operand; grad is autodiff, NOT finite difference)."""
    vg = jax.value_and_grad(chi2_pair)
    def f(p):
        v, g = vg(jnp.asarray(p)); return float(v), np.asarray(g, float)
    res = sopt.minimize(f, np.asarray(p_start, float), jac=True, method="L-BFGS-B", bounds=bounds,
                        options=dict(maxiter=300, ftol=1e-12, gtol=1e-10))
    return res.x, float(res.fun)


def _nongauss(dchi_true, dchi_gauss, X, Y, bfp):
    """Non-Gaussianity metrics on the region Delta chi^2_true < 9 (3 sigma)."""
    reg = dchi_true < 9.0
    resid = dchi_true - dchi_gauss
    out = dict(
        max_absdev=float(np.max(np.abs(resid[reg]))),
        rms_dev=float(np.sqrt(np.mean(resid[reg] ** 2))),
        frac_max=float(np.max(np.abs(resid[reg])) / 9.0),
        area_true_1s=float(np.sum(dchi_true < 1.0)),
        area_gauss_1s=float(np.sum(dchi_gauss < 1.0)),
    )
    out["area_ratio_1s"] = out["area_true_1s"] / max(out["area_gauss_1s"], 1.0)
    return out


def compute_pair(C, key, ng, obs_tag):
    name0, name1, tex0, tex1, r0, r1 = PAIRS[key]
    assemble, NOM = L.assembler([name0, name1])
    chi2_abs_pair = lambda p: C["chi2_abs"](assemble(p))       # NOT jitted (JBs must stay an operand)
    p_nom = np.array([float(NOM[name0]), float(NOM[name1])])
    bounds = [(max(CLIP[0], r0[0] - 0.2), min(CLIP[1], r0[1] + 0.2)),
              (max(CLIP[0], r1[0] - 0.2), min(CLIP[1], r1[1] + 0.2))]
    t0 = time.time()
    def log(m): print(f"  [{key}|{obs_tag}] [{time.time()-t0:5.1f}s] {m}", flush=True)

    # ---- BFP (from nominal) + exact Hessian ---------------------------------------------------- #
    bfp, chi2_min = _fit_bfp(chi2_abs_pair, p_nom, bounds)
    H = np.asarray(jax.hessian(chi2_abs_pair)(jnp.asarray(bfp)))
    # covariance V = 2 H^-1 (chi^2 convention: Delta chi^2 = (p-bfp)^T (H/2) (p-bfp))
    try:
        V = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        V = 2.0 * np.linalg.pinv(H)
    rail = [(abs(bfp[i] - bounds[i][0]) < 1e-6) or (abs(bfp[i] - bounds[i][1]) < 1e-6) for i in range(2)]
    log(f"BFP=({bfp[0]:.4f},{bfp[1]:.4f})  chi2/ndf={chi2_min/C['ndf']:.3f}  rail={rail}")
    log(f"H=\n{H}\nV=2H^-1=\n{V}")

    # ---- dense grid: exact Delta chi^2 (abs + shape) ------------------------------------------- #
    g0 = np.linspace(r0[0], r0[1], ng); g1 = np.linspace(r1[0], r1[1], ng)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")               # (ng,ng)
    chi2_grid = np.zeros((ng, ng)); chi2s_grid = np.zeros((ng, ng)); A_grid = np.zeros((ng, ng))
    both = C["chi2_both"]
    tstart = time.time()
    for i in range(ng):
        for j in range(ng):
            k = dict(NOM); k[name0] = float(G0[i, j]); k[name1] = float(G1[i, j])
            ca, cs, A = both(k)
            chi2_grid[i, j] = float(ca); chi2s_grid[i, j] = float(cs); A_grid[i, j] = float(A)
        if i % max(1, ng // 6) == 0:
            log(f"grid row {i+1}/{ng}  ({(time.time()-tstart)/((i+1)*ng)*1e3:.0f} ms/eval)")
    # BFP for the shape surface (its own min on the grid; the abs BFP is the primary one)
    chi2s_min = float(chi2s_grid.min())
    dchi_true = chi2_grid - chi2_min
    dchi_shape = chi2s_grid - chi2s_min

    # ---- Gaussian (Hessian) Delta chi^2 over the grid ------------------------------------------ #
    dP0 = G0 - bfp[0]; dP1 = G1 - bfp[1]
    Hh = 0.5 * H                                               # Delta chi^2 = dP^T (H/2) dP
    dchi_gauss = (Hh[0, 0] * dP0 ** 2 + 2 * Hh[0, 1] * dP0 * dP1 + Hh[1, 1] * dP1 ** 2)

    ng_metrics = _nongauss(dchi_true, dchi_gauss, G0, G1, bfp)
    log(f"non-Gaussianity: max|dev|={ng_metrics['max_absdev']:.2f} (frac {ng_metrics['frac_max']:.2f})  "
        f"RMS={ng_metrics['rms_dev']:.2f}  area(true/gauss @1s)={ng_metrics['area_ratio_1s']:.2f}")

    os.makedirs(OUTDIR, exist_ok=True)
    npz = f"{OUTDIR}/{key}_{obs_tag}.npz"
    np.savez(npz, key=key, obs_tag=obs_tag, name0=name0, name1=name1, tex0=tex0, tex1=tex1,
             g0=g0, g1=g1, chi2_grid=chi2_grid, chi2s_grid=chi2s_grid, A_grid=A_grid,
             dchi_true=dchi_true, dchi_shape=dchi_shape, dchi_gauss=dchi_gauss,
             bfp=bfp, chi2_min=chi2_min, chi2s_min=chi2s_min, H=H, V=V, rail=np.array(rail),
             p_nom=p_nom, ndf=C["ndf"], bounds=np.array(bounds),
             **{f"ng_{kk}": vv for kk, vv in ng_metrics.items()})
    log(f"saved -> {npz}")
    return npz


# ------------------------------------------------------------------ figure ------------------------------- #
def make_figure(npz):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz, allow_pickle=True)
    g0, g1 = d["g0"], d["g1"]; G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    dchi_true, dchi_gauss, dchi_shape = d["dchi_true"], d["dchi_gauss"], d["dchi_shape"]
    bfp, pnom = d["bfp"], d["p_nom"]; tex0, tex1 = str(d["tex0"]), str(d["tex1"])
    key, obs_tag = str(d["key"]), str(d["obs_tag"])
    levels = [1.0, 4.0, 9.0]                                    # 1/2/3 sigma (single-param Delta chi^2)
    lv2 = [2.30, 6.18, 11.83]                                   # 68/95/99% joint (2-dof)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    # -- panel 0: true Delta chi^2 heat + true contours + Hessian ellipse overlay --
    a = ax[0]
    pc = a.pcolormesh(G0, G1, np.clip(dchi_true, 0, 25), cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=a, fraction=0.046, label=r"$\Delta\chi^2$ (true, capped 25)")
    cs_t = a.contour(G0, G1, dchi_true, levels=levels, colors="w", linewidths=1.8)
    a.clabel(cs_t, fmt={1.0: "1", 4.0: "4", 9.0: "9"}, fontsize=8)
    cs_g = a.contour(G0, G1, dchi_gauss, levels=levels, colors="r", linewidths=1.4, linestyles="--")
    a.plot(pnom[0], pnom[1], "s", color="0.8", ms=9, mec="k", label="nominal")
    a.plot(bfp[0], bfp[1], "X", color="red", ms=12, mec="k", label="BFP")
    a.plot([], [], "w-", label=r"true $\Delta\chi^2$=1,4,9")
    a.plot([], [], "r--", label="Hessian ellipse")
    a.set_xlabel(tex0); a.set_ylabel(tex1); a.legend(fontsize=8, loc="best")
    a.set_title(f"True LLH vs Hessian ellipse  ({key})")

    # -- panel 1: profiled (min over partner) vs conditional (slice at BFP) 1D + shape-profiled --
    a = ax[1]
    # profile over p1 -> function of p0
    prof0 = dchi_true.min(axis=1); cond0 = dchi_true[:, np.argmin(np.abs(g1 - bfp[1]))]
    a.plot(g0, prof0, "C0-", lw=2, label=r"profiled $\Delta\chi^2(p_0)=\min_{p_1}$")
    a.plot(g0, cond0, "C0--", lw=1.4, label=r"conditional (slice at BFP $p_1$)")
    # gaussian 1D along p0 through bfp: use marginal V[0,0]
    V = d["V"]; sig0 = np.sqrt(abs(V[0, 0]))
    a.plot(g0, ((g0 - bfp[0]) / sig0) ** 2, "r:", lw=1.4, label=r"Gaussian ($\sigma$ from $V_{00}$)")
    a.axhline(1, color="0.6", lw=0.7, ls=":"); a.set_ylim(0, 12)
    a.set_xlabel(tex0); a.set_ylabel(r"$\Delta\chi^2$"); a.legend(fontsize=8)
    a.set_title("Profiled vs conditional vs Gaussian (along $p_0$)")

    # -- panel 2: non-Gaussianity residual (true - gauss) --
    a = ax[2]
    resid = dchi_true - dchi_gauss
    vmax = float(np.percentile(np.abs(resid[dchi_true < 12]), 99)) or 1.0
    pc = a.pcolormesh(G0, G1, resid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(pc, ax=a, fraction=0.046, label=r"$\Delta\chi^2_{\rm true}-\Delta\chi^2_{\rm Gauss}$")
    a.contour(G0, G1, dchi_true, levels=[1, 4, 9], colors="k", linewidths=0.8, alpha=0.5)
    a.plot(bfp[0], bfp[1], "X", color="k", ms=10)
    a.set_xlabel(tex0); a.set_ylabel(tex1)
    a.set_title(f"non-Gaussianity  max|dev|={float(d['ng_max_absdev']):.1f}  "
                f"area ratio@1$\\sigma$={float(d['ng_area_ratio_1s']):.2f}")

    rail = d["rail"]
    fig.suptitle(f"T2K CC0$\\pi$ [{obs_tag}] exact LLH surface — {tex0} $\\times$ {tex1}   "
                 f"(BFP $\\chi^2$/ndf={float(d['chi2_min'])/int(d['ndf']):.2f}"
                 f"{'  RAIL!' if bool(rail[0]) or bool(rail[1]) else ''})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("output/figures", exist_ok=True)
    out = f"output/figures/llh_surface_{key}_{obs_tag}.png"
    fig.savefig(out, dpi=130); plt.close(fig); print(f"wrote {out}", flush=True)


def main():
    obs = "dpt,dat"
    ng = 61
    if "--obs" in sys.argv: obs = sys.argv[sys.argv.index("--obs") + 1]
    if "--ng" in sys.argv: ng = int(sys.argv[sys.argv.index("--ng") + 1])
    # positional pair keys: drop flags AND their values
    skip = set()
    for fl in ("--obs", "--ng"):
        if fl in sys.argv:
            i = sys.argv.index(fl); skip.update({i, i + 1})
    args = [a for n, a in enumerate(sys.argv[1:], start=1)
            if n not in skip and not a.startswith("--")]
    keys = args if args else list(PAIRS)
    obs_list = obs.split(",")
    obs_tag = "".join(obs_list)
    t0 = time.time()
    print(f"[surface] pairs={keys} obs={obs_list} ng={ng}", flush=True)
    JBs, grids, aux = L.load_signal_bank()
    specs = [L.obs_binning(aux, o) for o in obs_list]
    C = L.make_chi2(JBs, grids, specs)
    for nm, cond, nf in zip(C["obs_names"], C["conds"], C["nfloored"]):
        print(f"[surface] cov[{nm}] cond={cond:.2e} floored={nf}", flush=True)
    for key in keys:
        npz = compute_pair(C, key, ng, obs_tag)
        make_figure(npz)
    print(f"[surface] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        main()
