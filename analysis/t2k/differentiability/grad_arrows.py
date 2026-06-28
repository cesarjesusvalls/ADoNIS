"""Per-bin JACOBIAN of a T2K CC0pi distribution w.r.t. the knobs, drawn as arrows on the histogram.

Compute J_i^k = d(dsigma/dx)_i / d theta_k at NOMINAL via autodiff of model_hist_full (averaged over the
frozen-walk replica bank to suppress MC noise).  Draw, per knob, an arrow from each bin center anchored at
the nominal bin value: UP if J_i>0 (red), DOWN if J_i<0 (blue), length proportional to |J_i|.

Two modes:
  python analysis/t2k/differentiability/grad_arrows.py [dpt|dat] all      # GRID, one panel per knob (default)
  python analysis/t2k/differentiability/grad_arrows.py [dpt|dat] M_A      # single knob (+ signed-bar panel)
  python analysis/t2k/differentiability/grad_arrows.py --plot-only <npz>

Tuple knobs are expanded: s_NN_elastic/s_NN_inelastic -> [pp]/[pn]/[nn]; pw_norm -> pw_norm[0..13].
In GRID mode arrow length uses ONE GLOBAL scale (comparable across knobs); each panel title carries max|J|.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

NREP = 4
_ISO = {0: "pp", 1: "pn", 2: "nn"}


def _specs(NOM):
    """Ordered (knob_name, component_idx|None, display_label, nominal_value).  pw_norm is EXCLUDED."""
    out = []
    for name, val in NOM.items():
        if name == "pw_norm":
            continue
        if isinstance(val, tuple):
            for i, vi in enumerate(val):
                lab = f"{name}[{_ISO[i]}]" if name.startswith("s_NN") else f"{name}[{i}]"
                out.append((name, i, lab, float(vi)))
        else:
            out.append((name, None, name, float(val)))
    return out


def _obs_binning(T, obs):
    """(edges, conv, obs_fn) for either observable -- read straight from the nuisance STV root files."""
    import uproot
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if obs == 'dpt' else 'dat'}Results.root")
    if obs == "dpt":
        edges = np.asarray(r["Result"].axis().edges()) * 1000.0
        conv = 1e-33 / 12.0 * 1000.0 * 1e38
    else:
        edges = np.asarray(r["Result"].axis().edges())
        conv = 1e-33 / 12.0 * 1e38
    return edges, conv, (T._dpt if obs == "dpt" else T._dat)


def run():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.argv = [sys.argv[0], "dpt"]                          # tune reads OBS at import; we bin BOTH explicitly
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, build_hv_sf, model_hist_full
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction

    T.NQE = T.NRES = int(os.environ.get("CC0PI_N", "30000"))
    NOM = nominal_knobs()
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log(f"Jacobian-arrow grid (dat+dpt), all knobs minus pw_norm.  N={T.NQE} NREP={NREP}")

    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    HV, SF = build_hv_sf(qe, res, sf, with_pw=False); log("hard-vertex amps2 records (no pw) + SF grids built")
    walks = [T.build_walk(jax.random.PRNGKey(50 + i), qe, qw, res, rw) for i in range(NREP)]
    log(f"{NREP} OBSERVABLE-INDEPENDENT walk replicas built (reused for BOTH observables)")

    SP = _specs(NOM)
    tup_names = sorted({n for n, idx, _, _ in SP if idx is not None})
    labels = [s[2] for s in SP]
    p0 = jnp.asarray([s[3] for s in SP])

    def assemble(p):
        k = dict(NOM)
        tup = {n: list(NOM[n]) for n in tup_names}
        for i, (name, idx, _, _) in enumerate(SP):
            if idx is None:
                k[name] = p[i]
            else:
                tup[name][idx] = p[i]
        for n in tup_names:
            k[n] = tuple(tup[n])
        return k

    npzs = []
    for obs in ("dat", "dpt"):
        edges, conv, ofn = _obs_binning(T, obs)
        bank = [T.bin_walk(W, edges=edges, obs=ofn) for W in walks]      # SAME walks, this observable's bins

        def hist_mean(p, bank=bank, edges=edges, conv=conv):
            kk = assemble(p)
            return jnp.mean(jnp.stack([model_hist_full(kk, R, HV, SF, edges, conv) for R in bank]), axis=0)

        h0 = np.asarray(hist_mean(p0))
        Js = np.asarray(jax.jacrev(hist_mean)(p0)).T                     # (nknob, nbins)
        log(f"{obs}: Jacobian done, {Js.shape[0]} knobs x {Js.shape[1]} bins; global max|J|={np.max(np.abs(Js)):.3e}")
        np.savez(f"/tmp/adonis_tune_runs/jac_grid_{obs}.npz", obs=obs, edges=edges, h0=h0, J=Js,
                 labels=np.array(labels))
        npzs.append(f"/tmp/adonis_tune_runs/jac_grid_{obs}.npz")
        make_grid(obs, edges, h0, Js, labels)
    build_pdf("output/figures/cc0pi_jacobian_arrows_all.pdf", npzs)


def _build_grid_fig(obs, edges, h0, Js, labels):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xsc = 1000.0 if obs == "dpt" else 1.0
    ctr = 0.5 * (edges[1:] + edges[:-1]) / xsc; xed = edges / xsc
    ARR = 0.20                                              # EACH subplot's longest arrow = ARR * y-axis span
    smax = np.max(np.abs(Js), axis=1)                       # per-subplot max|J| (own physical scale)
    hmax = float(h0.max())
    yhi, ylo = hmax * 1.3, -0.05 * hmax                    # fixed-point: arrow length <-> axis span (no clip)
    for _ in range(200):
        span = yhi - ylo
        sc = np.where(smax > 0, ARR * span / np.where(smax > 0, smax, 1.0), 0.0)  # per-panel scale
        dys = Js * sc[:, None]                              # each panel: longest arrow = ARR*span
        tops = h0[None, :] + np.where(Js > 0, dys, 0.0)
        bots = h0[None, :] + np.where(Js < 0, dys, 0.0)
        nyhi = max(hmax, float(tops.max())) + 0.05 * span
        nylo = min(0.0, float(bots.min())) - 0.05 * span
        if abs(nyhi - yhi) < 1e-12 and abs(nylo - ylo) < 1e-12:
            break
        yhi, ylo = nyhi, nylo
    span = yhi - ylo
    sc = np.where(smax > 0, ARR * span / smax, 0.0)
    dys = Js * sc[:, None]                                  # => per-panel longest arrow = ARR*span exactly

    n = len(labels); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    XL = r"$\delta p_T$ [GeV/c]" if obs == "dpt" else r"$\delta\alpha_T$ [rad]"
    for j, lab in enumerate(labels):
        ax = axes[j]; J = Js[j]; dy = dys[j]
        ax.fill_between(xed, np.append(h0, h0[-1]), step="post", color="0.92", zorder=0)
        ax.step(xed, np.append(h0, h0[-1]), where="post", color="0.7", lw=1.0, zorder=1)
        col = np.where(J >= 0, "#c0392b", "#2471a3")
        ax.quiver(ctr, h0, np.zeros_like(dy), dy, angles="xy", scale_units="xy", scale=1.0,
                  color=col, width=0.016, headwidth=4, headlength=5, zorder=3)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(f"{lab}  |J|$_\\mathrm{{max}}$={np.max(np.abs(J)):.1e}", fontsize=8)
        ax.set_ylim(ylo, yhi); ax.tick_params(labelsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    for j in range(n):
        if j >= n - ncol:
            axes[j].set_xlabel(XL, fontsize=8)
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"d$\sigma$/dx", fontsize=8)
    fig.suptitle(f"Per-bin Jacobian $\\partial(\\mathrm{{d}}\\sigma/\\mathrm{{d}}x)_i/\\partial\\theta$ on {obs} "
                 f"— all knobs (up=+ red / down=$-$ blue; EACH subplot scaled so its longest arrow = "
                 f"{ARR:.1f}$\\times$y-span — sizes NOT comparable across panels, see |J|$_\\mathrm{{max}}$)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


def make_grid(obs, edges, h0, Js, labels):
    fig = _build_grid_fig(obs, edges, h0, Js, labels)
    os.makedirs("output/figures", exist_ok=True)
    FIG = f"output/figures/cc0pi_jacobian_arrows_grid_{obs}.png"
    fig.savefig(FIG, dpi=120); print(f"wrote {FIG}", flush=True)


def build_pdf(out_path, npz_paths):
    """Assemble one multi-page PDF (one page per observable npz) of the all-knob Jacobian grids."""
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with PdfPages(out_path) as pdf:
        for p in npz_paths:
            d = np.load(p, allow_pickle=True)
            fig = _build_grid_fig(str(d["obs"]), d["edges"], d["h0"], d["J"], list(d["labels"]))
            pdf.savefig(fig, dpi=120)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    if "--pdf" in sys.argv:                                  # --pdf <out.pdf> <npz1> <npz2> ...
        i = sys.argv.index("--pdf"); build_pdf(sys.argv[i + 1], sys.argv[i + 2:])
    elif "--plot-only" in sys.argv:
        d = np.load(sys.argv[sys.argv.index("--plot-only") + 1], allow_pickle=True)
        make_grid(str(d["obs"]), d["edges"], d["h0"], d["J"], list(d["labels"]))
    else:
        run()
