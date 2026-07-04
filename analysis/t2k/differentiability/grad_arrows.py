"""WALK-TIME knob-Jacobian figures for the T2K CC0pi model (autodiff of model_hist_full over freshly
sampled proposal + pool-walk replicas -- use this when you need Jacobians at settings the frozen event bank
does not cover; for bank-based exact variations use bank_arrows.py).

Three modes (merged from the former grad_arrows.py / grad_2d.py / grad_all.py):

  1D arrow grids (default)  -- per-bin J = d(dsigma/dx)_i/dtheta_k on dpt+dat, drawn as arrows per knob:
     CC0PI_N=30000 python analysis/t2k/differentiability/grad_arrows.py
     python analysis/t2k/differentiability/grad_arrows.py --plot-only /tmp/adonis_tune_runs/jac_grid_dpt.npz

  2D heatmap (--2d)         -- per-knob per-cell Jacobian over the (dpt, dat) plane:
     CC0PI_N=30000 python analysis/t2k/differentiability/grad_arrows.py --2d
     python analysis/t2k/differentiability/grad_arrows.py --2d --plot-only /tmp/adonis_tune_runs/jac2d.npz [--relative]

  high-stats driver (--hi-stats) -- CHUNKED streaming accumulation of BOTH the 1D and 2D Jacobians (peak
  memory ~ one chunk), cached to output/jac_cache so figures re-render without recompute:
     CC0PI_N=1000000 CHUNK=80000 NREP=1 python analysis/t2k/differentiability/grad_arrows.py --hi-stats
     python analysis/t2k/differentiability/grad_arrows.py --hi-stats --replot

Knob enumeration comes from full_knobs.knob_specs (pw_norm and the dead sscat excluded); tuple knobs are
expanded (s_NN_elastic -> [pp]/[pn]/[nn]).  All runs need the ../nuisance T2K data files for the binning.
"""
import os, sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from analysis.t2k.differentiability.full_knobs import knob_specs

NREP = int(os.environ.get("NREP", "4"))
CACHE = "output/jac_cache"
PDF_1D = "output/figures/cc0pi_jacobian_arrows_all.pdf"


def _specs(NOM):
    """Back-compat alias -- the knob table lives in full_knobs.knob_specs (single source of truth)."""
    return knob_specs(NOM)


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


def _assemble_fn(NOM, SP):
    """theta-vector -> knob dict (tuple components rebuilt); shared by all modes."""
    tup_names = sorted({n for n, idx, _, _ in SP if idx is not None})

    def assemble(p):
        k = dict(NOM); tup = {n: list(NOM[n]) for n in tup_names}
        for i, (name, idx, _, _) in enumerate(SP):
            (k.__setitem__(name, p[i]) if idx is None else tup[name].__setitem__(idx, p[i]))
        for n in tup_names:
            k[n] = tuple(tup[n])
        return k
    return assemble


# ================================================================ 1D arrow grids (former grad_arrows) === #
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

    SP = knob_specs(NOM)
    labels = [s[2] for s in SP]
    p0 = jnp.asarray([s[3] for s in SP])
    assemble = _assemble_fn(NOM, SP)

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
    build_pdf(PDF_1D, npzs)


def _build_grid_fig(obs, edges, h0, Js, labels, xlabel=None, xsc=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if xsc is None:
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
    sc = np.where(smax > 0, ARR * span / np.where(smax > 0, smax, 1.0), 0.0)
    dys = Js * sc[:, None]                                  # => per-panel longest arrow = ARR*span exactly

    n = len(labels); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    XL = xlabel if xlabel is not None else (r"$\delta p_T$ [GeV/c]" if obs == "dpt" else r"$\delta\alpha_T$ [rad]")
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


# ================================================================ 2D heatmap (former grad_2d) =========== #
def binning2d(obs):
    import uproot
    r = uproot.open(f"../nuisance/data/T2K/CC0pi/STV/{'dpt' if obs == 'dpt' else 'dat'}Results.root")
    e = np.asarray(r["Result"].axis().edges())
    return e * 1000.0 if obs == "dpt" else e               # dpt in MeV, dat in rad


def run_2d():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, build_hv_sf, _hv_qe, _hv_res, _fsi
    from adonis.analysis.sf_reweight import sf_reweight
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction

    T.NQE = T.NRES = int(os.environ.get("CC0PI_N", "30000"))
    NOM = nominal_knobs()
    edpt, edat = binning2d("dpt"), binning2d("dat")
    n1, n2 = len(edpt) - 1, len(edat) - 1
    ej1, ej2 = jnp.asarray(edpt), jnp.asarray(edat)
    area = np.outer(np.diff(edpt), np.diff(edat))           # MeV * rad
    CONV2D = 1e-33 / 12.0 * 1e38 * 1000.0                   # -> 1e-38 cm^2/(GeV/c . rad)/nucleon
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    log(f"2D Jacobian heatmap (dpt x dat = {n1}x{n2})  N={T.NQE} NREP={NREP}")

    qe, qw, res, rw = T.build_proposal(); log("proposal sampled")
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    HV, SF = build_hv_sf(qe, res, sf, with_pw=False); log("hard-vertex amps2 records (no pw) + SF grids built")
    walks = [T.build_walk(jax.random.PRNGKey(50 + i), qe, qw, res, rw) for i in range(NREP)]
    log(f"{NREP} walk replicas built")

    def bin2d(W):
        qd, qa = T._dpt(W["q_kmu"], W["q_lead"]), T._dat(W["q_kmu"], W["q_lead"])
        rd, ra = T._dpt(W["r_kmu"], W["r_lead"]), T._dat(W["r_kmu"], W["r_lead"])
        qi = jnp.clip(jnp.searchsorted(ej1, qd) - 1, 0, n1 - 1); qj = jnp.clip(jnp.searchsorted(ej2, qa) - 1, 0, n2 - 1)
        ri = jnp.clip(jnp.searchsorted(ej1, rd) - 1, 0, n1 - 1); rj = jnp.clip(jnp.searchsorted(ej2, ra) - 1, 0, n2 - 1)
        return dict(q_flat=qi * n2 + qj, q_keep=T._sel(W["q_kmu"], W["q_lead"]), q_w0=W["q_w0"], q_rec=W["q_rec"],
                    r_flat=ri * n2 + rj, r_keep=T._sel(W["r_kmu"], W["r_lead"]), r_w0=W["r_w0"], r_rec=W["r_rec"])
    banks = [bin2d(W) for W in walks]

    qsf_args = (SF["grids"], SF["qe_pmag"], SF["qe_erem"]); rsf_args = (SF["grids"], SF["res_pmag"], SF["res_erem"])
    SP = knob_specs(NOM); labels = [s[2] for s in SP]; p0 = jnp.asarray([s[3] for s in SP])
    assemble = _assemble_fn(NOM, SP)

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
    Jflat = np.asarray(jax.jacrev(hist2d_flat)(p0))                  # (n1*n2, nknob)
    log(f"2D Jacobian done: {Jflat.shape[1]} knobs x {n1*n2} cells")
    norm2d = (h0 / area) * CONV2D
    J2d = (Jflat.T.reshape(len(labels), n1, n2) / area[None]) * CONV2D    # (nknob, n1, n2)

    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    np.savez("/tmp/adonis_tune_runs/jac2d.npz", edpt=edpt, edat=edat, norm2d=norm2d, J2d=J2d,
             labels=np.array(labels))
    make_figure_2d("/tmp/adonis_tune_runs/jac2d.npz")


def make_figure_2d(npz_path, relative=False, rate_frac=0.01):
    """Per-knob 2D sensitivity heatmap.  relative=False -> ABSOLUTE d(d2sigma)/dtheta; relative=True ->
    FRACTIONAL d(ln sigma)/dtheta = J2d/norm2d (cells below rate_frac*max(norm2d) masked: relative is just
    noise where there is ~no rate).  Per-panel symmetric scale (robust 99th pct for the relative case)."""
    d = np.load(npz_path, allow_pickle=True)
    edpt, edat = d["edpt"] / 1000.0, d["edat"]              # dpt back to GeV/c for axes
    norm2d, J2d, labels = d["norm2d"], d["J2d"], list(d["labels"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if relative:
        mask = norm2d <= rate_frac * float(norm2d.max())    # drop low-rate cells (noisy ratio)
        denom = np.where(mask, 1.0, norm2d)
        field = J2d / denom[None]                            # d(ln sigma)/dtheta
        sym = "$\\partial(\\ln\\sigma)/\\partial\\theta$"; unit = "[/unit knob]"; tag = "relative"
        robust = True
    else:
        mask = norm2d <= 0                                   # kinematically empty cells
        field = J2d
        sym = "$\\partial(\\mathrm{d}^2\\sigma)/\\partial\\theta$"; unit = ""; tag = "absolute"
        robust = False
    n = len(labels); ncol = 6; nrow = int(np.ceil((n + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    X, Y = np.meshgrid(edpt, edat, indexing="ij")

    nm = np.ma.masked_where(mask, norm2d)
    a0 = axes[0]
    pc = a0.pcolormesh(X, Y, nm, cmap="Greys", shading="flat")
    a0.set_title("nominal d$^2\\sigma$/d$\\delta p_T$d$\\delta\\alpha_T$", fontsize=8)
    a0.set_ylabel(r"$\delta\alpha_T$ [rad]", fontsize=8); fig.colorbar(pc, ax=a0, fraction=0.046)

    for j, lab in enumerate(labels):
        ax = axes[j + 1]; F = np.ma.masked_where(mask, field[j])
        vals = np.abs(field[j][~mask])
        vmax = float(np.percentile(vals, 99) if robust and vals.size else (np.max(vals) if vals.size else 1.0)) or 1.0
        pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        ax.set_title(f"{lab}  max={vmax:.1e}", fontsize=8)
        fig.colorbar(pc, ax=ax, fraction=0.046)
    for j in range(n + 1, len(axes)):
        axes[j].axis("off")
    for k in range(ncol):
        idx = (nrow - 1) * ncol + k
        if idx < n + 1:
            axes[idx].set_xlabel(r"$\delta p_T$ [GeV/c]", fontsize=8)
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"$\delta\alpha_T$ [rad]", fontsize=8)
    extra = f"; cells <{rate_frac:.0%} of peak rate masked" if relative else ""
    fig.suptitle(f"Per-knob 2D {tag} sensitivity {sym} {unit} over "
                 f"$(\\delta p_T,\\,\\delta\\alpha_T)$  (red=+ / blue=$-$; per-panel symmetric scale{extra})",
                 fontsize=12)
    os.makedirs("output/figures", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    stem = "cc0pi_jacobian_2d_relative" if relative else "cc0pi_jacobian_2d"
    for ext in ("png", "pdf"):
        Fn = f"output/figures/{stem}.{ext}"; fig.savefig(Fn, dpi=120); print(f"wrote {Fn}", flush=True)


# ================================================================ high-stats driver (former grad_all) === #
def _render_cached():
    """(Re)build all figures from the cached npz -- the plotting-only path."""
    npzs = []
    for o in ("dat", "dpt"):
        d = np.load(f"{CACHE}/jac_grid_{o}.npz", allow_pickle=True)
        make_grid(o, d["edges"], d["h0"], d["J"], list(d["labels"]))
        npzs.append(f"{CACHE}/jac_grid_{o}.npz")
    build_pdf(PDF_1D, npzs)
    make_figure_2d(f"{CACHE}/jac2d.npz")


def run_hi_stats():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _replot = "--replot" in sys.argv
    sys.argv = [sys.argv[0], "dpt"]                          # tune reads OBS at import; we bin BOTH explicitly
    from analysis.t2k.differentiability import tune as T
    from analysis.t2k.differentiability.full_knobs import (nominal_knobs, build_hv_sf, model_hist_full,
                                                           _hv_qe, _hv_res, _fsi)
    from adonis.analysis.sf_reweight import sf_reweight
    from adonis.workflow.materials import resolve_targets
    from adonis.xsec.spectral import SpectralFunction
    from adonis.xsec import qe_xsec, res_xsec

    if _replot:
        _render_cached(); return

    N_TOTAL = int(os.environ.get("CC0PI_N", "120000"))
    CHUNK = min(int(os.environ.get("CHUNK", str(N_TOTAL))), N_TOTAL)
    n_chunks = int(np.ceil(N_TOTAL / CHUNK))
    sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
    NOM = nominal_knobs()
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    log(f"hi-stats: N_total={N_TOTAL} in {n_chunks} chunk(s) of {CHUNK}, NREP={NREP}  "
        f"(primary events={CHUNK*n_chunks}, cascades={CHUNK*n_chunks*NREP})")

    SP = knob_specs(NOM)
    labels = [s[2] for s in SP]; p0 = jnp.asarray([s[3] for s in SP]); nk = len(labels)
    assemble = _assemble_fn(NOM, SP)

    OBS1D = {o: _obs_binning(T, o) for o in ("dat", "dpt")}
    edpt, edat = binning2d("dpt"), binning2d("dat")
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
    _render_cached()
    log("figures rendered")


if __name__ == "__main__":
    os.makedirs("/tmp/adonis_tune_runs", exist_ok=True)
    if "--hi-stats" in sys.argv:
        run_hi_stats()
    elif "--2d" in sys.argv:
        if "--plot-only" in sys.argv:
            make_figure_2d(sys.argv[sys.argv.index("--plot-only") + 1], relative="--relative" in sys.argv)
        else:
            run_2d()
    elif "--plot-only" in sys.argv:
        d = np.load(sys.argv[sys.argv.index("--plot-only") + 1], allow_pickle=True)
        make_grid(str(d["obs"]), d["edges"], d["h0"], d["J"], list(d["labels"]))
    else:
        run()
