"""3D LLH slice: map the exact T2K CC0pi chi^2 over a dense 3-parameter cube, and read off the degeneracy
structure that a single FD-Hessian-at-BFP cannot show -- the 3x3 exact-Hessian eigen-spectrum (stiff vs
flat directions) contrasted against the true PROFILED 2D projections (min over the 3rd param).

Triples (keys):
  qe_trio  : M_A_qe x axial_strength x qe_norm      (all scale the QE axial rate -> curved 3D valley)
  fsi_trio : sabs x s_piN_elastic x s_piN_cex        (inside the flagged exact pion-FSI flat direction:
             equal rescale of {sabs, s_piN_elastic, s_piN_cex, s_conv} is a per-event invariance -> the
             3x3 Hessian is near-singular; the true surface has a genuine flat plane)

  python -u scripts/cc0pi_llh_3d.py [qe_trio|fsi_trio] [--ng 25]
  python -u scripts/cc0pi_llh_3d.py --plot-only /tmp/adonis_llh/<key>_3d.npz
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.optimize as sopt

import cc0pi_llh_lib as L

OUTDIR = "/tmp/adonis_llh"
CLIP = (0.3, 3.0)

TRIPLES = {
    # M_A_qe x axial x res_norm (the task's literal suggestion): the QE-axial banana in (M_A_qe, axial) plus
    # a soft res_norm axis -- interior BFP (no three-way rate degeneracy, unlike a QE-only-norm trio).
    "ma_axial_resnorm": (["M_A_qe", "axial_strength", "res_norm"],
                 [r"$M_A^{\rm QE}$", r"$g_A^{\rm QE}$", "RES norm"],
                 [(0.65, 1.70), (0.50, 1.45), (0.30, 2.70)]),
    # sabs x s_piN_elastic x s_piN_cex: 3 of the 4 knobs in the flagged EXACT pion-FSI flat direction
    # (equal rescale of {sabs, s_piN_elastic, s_piN_cex, s_conv} is a per-event invariance) -> near-singular H.
    "fsi_trio": (["sabs", "s_piN_elastic", "s_piN_cex"],
                 [r"$\sigma_{\pi\,\rm abs}$", r"$\sigma_{\pi N\,\rm el}$", r"$\sigma_{\pi N\,\rm cex}$"],
                 [(0.40, 2.20), (0.35, 2.60), (0.35, 2.60)]),
}


def _fit_bfp(chi2_fn, p0, bounds):
    vg = jax.value_and_grad(chi2_fn)
    def f(p):
        v, g = vg(jnp.asarray(p)); return float(v), np.asarray(g, float)
    r = sopt.minimize(f, np.asarray(p0, float), jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(maxiter=400, ftol=1e-12, gtol=1e-10))
    return r.x, float(r.fun)


def compute(C, key, ng):
    names, texs, rngs = TRIPLES[key]
    assemble, NOM = L.assembler(names)
    chi2 = lambda p: C["chi2_abs"](assemble(p))
    p_nom = np.array([float(NOM[n]) for n in names])
    bounds = [(max(CLIP[0], r[0] - 0.2), min(CLIP[1], r[1] + 0.2)) for r in rngs]
    t0 = time.time()
    def log(m): print(f"  [{key}] [{time.time()-t0:5.1f}s] {m}", flush=True)

    bfp, chi2_min = _fit_bfp(chi2, p_nom, bounds)
    H = np.asarray(jax.hessian(chi2)(jnp.asarray(bfp)))
    rail = [(abs(bfp[i] - bounds[i][0]) < 1e-6) or (abs(bfp[i] - bounds[i][1]) < 1e-6) for i in range(3)]
    log(f"BFP={np.round(bfp,4)}  chi2/ndf={chi2_min/C['ndf']:.3f}  rail={rail}")
    # normalized-Hessian eigen-spectrum (stiff vs flat directions)
    dg = np.sqrt(np.maximum(np.diag(H), 1e-300)); Hn = H / np.outer(dg, dg)
    ew, ev = np.linalg.eigh(Hn)
    log(f"H=\n{H}")
    log(f"cond(H)={np.linalg.cond(H):.3e}   normalized-H eigenvalues (small=flat): {np.round(ew,4)}")
    for j in range(3):
        comp = " ".join(f"{ev[t,j]:+.2f}*{names[t]}" for t in np.argsort(-np.abs(ev[:, j])))
        log(f"    lam={ew[j]:.3e}  dir: {comp}")

    axes = [np.linspace(r[0], r[1], ng) for r in rngs]
    cube = np.zeros((ng, ng, ng))
    both = C["chi2_abs"]
    tstart = time.time()
    for i in range(ng):
        ki = float(axes[0][i])
        for j in range(ng):
            kj = float(axes[1][j])
            for l in range(ng):
                k = dict(NOM); k[names[0]] = ki; k[names[1]] = kj; k[names[2]] = float(axes[2][l])
                cube[i, j, l] = float(both(k))
        if i % max(1, ng // 8) == 0:
            log(f"cube slab {i+1}/{ng}  ({(time.time()-tstart)/((i+1)*ng*ng)*1e3:.0f} ms/eval)")
    dcube = cube - chi2_min

    os.makedirs(OUTDIR, exist_ok=True)
    npz = f"{OUTDIR}/{key}_3d.npz"
    np.savez(npz, key=key, names=np.array(names), texs=np.array(texs),
             a0=axes[0], a1=axes[1], a2=axes[2], dcube=dcube, cube=cube,
             bfp=bfp, chi2_min=chi2_min, H=H, Hn=Hn, eigval=ew, eigvec=ev,
             rail=np.array(rail), p_nom=p_nom, ndf=C["ndf"], cond=np.linalg.cond(H))
    log(f"saved -> {npz}")
    return npz


def make_figure(npz):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(npz, allow_pickle=True)
    a = [d["a0"], d["a1"], d["a2"]]; texs = [str(x) for x in d["texs"]]
    dcube = d["dcube"]; bfp = d["bfp"]; H = d["H"]; ew = d["eigval"]; ev = d["eigvec"]
    names = [str(x) for x in d["names"]]; key = str(d["key"])
    pairs = [(0, 1), (0, 2), (1, 2)]
    levels = [1.0, 4.0, 9.0]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    for pi, (u, v) in enumerate(pairs):
        axu = pi
        # profiled 2D: min over the remaining axis (min over axis=rem leaves the two kept axes in order)
        rem = ({0, 1, 2} - {u, v}).pop()
        P = dcube.min(axis=rem)
        U, Vv = np.meshgrid(a[u], a[v], indexing="ij")
        aa = ax[axu]
        pc = aa.pcolormesh(U, Vv, np.clip(P, 0, 20), cmap="viridis", shading="auto")
        cs = aa.contour(U, Vv, P, levels=levels, colors="w", linewidths=1.6)
        aa.clabel(cs, fmt={1: "1", 4: "4", 9: "9"}, fontsize=7)
        # projected Hessian ellipse: submatrix inverse (conditional) vs full-inverse (profiled)
        sub = np.array([[H[u, u], H[u, v]], [H[v, u], H[v, v]]])
        try:
            Vfull = 2.0 * np.linalg.inv(H); Vp = np.array([[Vfull[u, u], Vfull[u, v]], [Vfull[v, u], Vfull[v, v]]])
            wv, vv = np.linalg.eigh(np.linalg.inv(Vp))       # profiled ellipse from marginal cov
            th = np.linspace(0, 2 * np.pi, 120)
            for Ln in (1.0, 4.0):
                xy = vv @ (np.sqrt(Ln / np.maximum(wv, 1e-12))[:, None] * np.array([np.cos(th), np.sin(th)]))
                aa.plot(bfp[u] + xy[0], bfp[v] + xy[1], "r--", lw=1.1)
        except np.linalg.LinAlgError:
            pass
        aa.plot(bfp[u], bfp[v], "X", color="red", ms=11, mec="k")
        aa.set_xlabel(texs[u]); aa.set_ylabel(texs[v])
        aa.set_title(f"profiled $\\Delta\\chi^2$ (min over {texs[rem]})", fontsize=9)
        fig.colorbar(pc, ax=aa, fraction=0.046)
    # eigen-spectrum panel
    aa = ax[3]
    order = np.argsort(ew)
    aa.bar(range(3), ew[order], color=["C3", "C1", "C0"])
    for r, j in enumerate(order):
        top = np.argsort(-np.abs(ev[:, j]))
        lab = "\n".join(f"{ev[t,j]:+.2f} {names[t]}" for t in top)
        aa.text(r, ew[j] + 0.02 * ew.max(), lab, ha="center", va="bottom", fontsize=6.5)
    aa.set_yscale("log"); aa.set_xticks(range(3)); aa.set_xticklabels(["flat", "mid", "stiff"])
    aa.set_ylabel("normalized-Hessian eigenvalue")
    aa.set_title(f"eigen-spectrum  cond(H)={float(d['cond']):.1e}", fontsize=9)
    rail = d["rail"]
    fig.suptitle(f"T2K CC0$\\pi$ [dpt+dat] 3D LLH slice — {texs[0]} $\\times$ {texs[1]} $\\times$ {texs[2]}  "
                 f"(BFP $\\chi^2$/ndf={float(d['chi2_min'])/int(d['ndf']):.2f}"
                 f"{'  RAIL!' if any(bool(x) for x in rail) else ''})  "
                 f"— red dashed = profiled Hessian ellipse", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("output/figures", exist_ok=True)
    out = f"output/figures/llh_3d_{key}.png"
    fig.savefig(out, dpi=130); plt.close(fig); print(f"wrote {out}", flush=True)


def main():
    ng = 25
    if "--ng" in sys.argv: ng = int(sys.argv[sys.argv.index("--ng") + 1])
    keys = [a for a in sys.argv[1:] if a in TRIPLES]
    if not keys: keys = list(TRIPLES)
    t0 = time.time()
    print(f"[3d] triples={keys} ng={ng}", flush=True)
    JBs, grids, aux = L.load_signal_bank()
    specs = [L.obs_binning(aux, o) for o in ("dpt", "dat")]
    C = L.make_chi2(JBs, grids, specs)
    for key in keys:
        npz = compute(C, key, ng)
        make_figure(npz)
    print(f"[3d] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        make_figure(sys.argv[sys.argv.index("--plot-only") + 1])
    else:
        main()
