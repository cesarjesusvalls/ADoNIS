"""ONE render module for every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

All figure code lives here; the diversity lives in the YAML specs next to this file, and make.py is the
only entry point.  A spec's `render:` key picks the render function:

  multiobs  pure selection histograms  -> adonis.workflow.analyze.run_analysis under the paper style
            (figs 7/8/9/12 + appendix A1/A2 -- nothing but a config).
  panels    compute a panel set, then adonis.workflow.plotting.make_figure (the shared grid-of-
            chi2_ratio_panel builder).  figs 1, 10, 11, 4-6: they differ ONLY in the COMPUTE that fills
            the panels (`compute:` selects it) and the layout kwargs (`layout:` in the spec).
  sigma_channels / beam_sigma / dcc   the three genuinely bespoke figures (2, 3, 13): multi-series
            overlays / angular samplers that are NOT one-observable panels, so they keep a dedicated
            function -- but still here, still config-driven, still one entry point.

Shared drawing (chi2_ratio_panel, make_figure) already lives in adonis.workflow.plotting; this module is
the paper-side COMPUTE + the thin dispatch onto it.
"""
import glob
import sys
from pathlib import Path

import numpy as np

# Repo root = the first ancestor that contains the `adonis` package (marker-based, not a hand-counted
# .parents[N]).  make.py resolves it the same way; the two must not drift.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))

from adonis.workflow.analyze import run_analysis                 # noqa: E402
from adonis.workflow.config import load_analysis_config   # noqa: E402
from adonis.workflow import selection as SG                      # noqa: E402
from adonis.workflow.plotting import make_figure, chi2_ratio_panel   # noqa: E402
from analysis.paper import style                                 # noqa: E402
from analysis.paper import plotcache                             # noqa: E402


def _p(spec):
    return dict((spec or {}).get("params", {}) or {})


def _rel(path):
    return str(ROOT / path)


# =============================================================== shared panel packer (all `panels` figs)
def _assemble(panels):
    """panels: list of {key, edges, label, ado:(values,w,chan), ref:(values,w,chan)}.
    Pack into the (specs, ado_sel, ref_sel) make_figure wants: one shared 'w'/'chan' per side, each key
    NaN-padded outside its own panel (make_figure keys every observable off one weight vector).  This is
    the single home of the concatenate+NaN-pad idiom the sliced and electron computes both need."""
    specs = [(p["key"], np.asarray(p["edges"], float), p["label"]) for p in panels]

    def side(which):
        w = np.concatenate([p[which][1] for p in panels])
        sel = {"w": w, "chan": np.concatenate([p[which][2] for p in panels])}
        off = np.cumsum([0] + [len(p[which][1]) for p in panels])
        for i, p in enumerate(panels):
            v = np.full(len(w), np.nan); v[off[i]:off[i + 1]] = p[which][0]; sel[p["key"]] = v
        return sel
    return specs, side("ado"), side("ref")


def _panel_kw(p):
    """Paper palette for a `panels` figure.  A single-line figure (params `breakdown: false`) draws the
    one series in the IBM blue (C_QE), NOT the QE/RES-total pink (C_TOTAL) -- pink reads as 'total of the
    components shown', which is wrong when no components are drawn.  This is the one home of that rule."""
    over = {"ratio_yticks": p.get("ratio_yticks", [0.8, 1.0, 1.2])}
    if not p.get("breakdown", True):
        over["total_color"] = style.C_QE
    return style.panel_kw(**over)


def _suppress_breakdown(p, ado_sel, ref_sel):
    """When params `breakdown: false`, zero the QE/RES channel so make_figure draws a single line (and its
    legend_fn, which keys off having parts, draws no legend)."""
    if not p.get("breakdown", True):
        ado_sel["chan"] = np.zeros(len(ado_sel["w"]), int)
        ref_sel["chan"] = np.zeros(len(ref_sel["w"]), int)


# =============================================================== COMPUTE: sliced selection (figs 10, 11)
def _compute_sliced(spec):
    """Selection histogrammed in SLICES of a second variable -> (specs, ado_sel, ref_sel, layout).
    Config: params.slice_by/slice_edges (the slice variable + its edges), params.obs/obs_edges (the
    histogrammed observable + per-slice edges, or one edge set), params.nc (NC selection)."""
    p = _p(spec)
    cfg = load_analysis_config(_rel(spec["_path"]))
    nc = bool(p.get("nc", False))
    getsel = (SG.bank_signal_nc, SG.oracle_signal_nc) if nc else (SG.bank_signal, SG.oracle_signal)
    ado = getsel[0](cfg.inputs["adonis_bank"][0], cfg.signal)
    ref = getsel[1](cfg.inputs["reference"][0], cfg.signal)

    slice_by = p["slice_by"]; s_edges = np.asarray(p["slice_edges"], float)
    obs = p["obs"]; nslice = len(s_edges) - 1
    # obs_edges: one uniform edge list (linspace lo,hi,n) OR one explicit list per slice; scaled by edge_scale
    if "obs_linspace" in p:
        lo_, hi_, n_ = p["obs_linspace"]; oe = [np.linspace(lo_, hi_, int(n_))] * nslice
    else:
        raw = p["obs_edges"]
        oe = [np.asarray(raw, float)] * nslice if np.ndim(raw[0]) == 0 else [np.asarray(e, float) for e in raw]
    edge_scale = float(p.get("edge_scale", 1.0))     # GeV->MeV etc. on the OBSERVABLE bin edges (not values)
    slice_scale = float(p.get("slice_scale", 1.0))   # e.g. rad->deg on the SLICE variable
    stex = p.get("slice_tex", slice_by)

    sv_a = np.asarray(ado[slice_by], float) * slice_scale
    sv_r = np.asarray(ref[slice_by], float) * slice_scale
    va = np.asarray(ado[obs], float); vr = np.asarray(ref[obs], float)
    panels = []
    for i in range(nslice):
        lo, hi = s_edges[i], s_edges[i + 1]
        labels = p.get("labels")
        label = labels[i] if labels else rf"{p.get('obs_label', obs)},  ${lo:g}<{stex}<{hi:g}$"
        ka = (sv_a >= lo) & (sv_a < hi); kr = (sv_r >= lo) & (sv_r < hi)
        panels.append({"key": f"{obs}_s{i}", "edges": np.asarray(oe[i], float) * edge_scale, "label": label,
                       "ado": (va[ka], np.asarray(ado["w"], float)[ka], np.asarray(ado["chan"])[ka]),
                       "ref": (vr[kr], np.asarray(ref["w"], float)[kr], np.asarray(ref["chan"])[kr])})
    specs, ado_sel, ref_sel = _assemble(panels)
    _suppress_breakdown(p, ado_sel, ref_sel)
    layout = dict(title=cfg.title, ratio_band=tuple(cfg.ratio_band), ratio_ylim=tuple(cfg.ratio_ylim),
                  panel_kw=_panel_kw(p))
    return specs, ado_sel, ref_sel, layout


# =============================================================== COMPUTE: electron (e,e') beam (figs 1, 4-6)
# One config-driven compute for both electron figures.  The (e,e') physics (omega, E_QE, E_cal, P_T +
# leading proton) lives in selection.ele_signal / ele_oracle_signal; here we only pick, per params.panels
# entry, an observable under a topology (incl / 0pi / 1p0pi), fetch the reduction (memoized per bank/ref),
# and hand the panels to _assemble.  fig01 = omega across two nuclei (per-panel bank); fig0456 = three
# observables + topologies on one bank.  Nothing electron-specific is hardcoded -- cuts/paths/edges are YAML.
_NUC_TEX = {"C": r"$^{12}$C", "Ar": r"$^{40}$Ar"}   # also used by render_beam_sigma (fig 3)
_TOPO = {"incl": lambda o: np.ones(len(o["w"]), bool),
         "0pi": lambda o: o["npi"] == 0,
         "1p0pi": lambda o: (o["npi"] == 0) & (o["nprot"] == 1)}


def _edges_of(ps):
    if "edges" in ps:
        return np.asarray(ps["edges"], float)
    lo, hi, n = ps["linspace"]
    return np.linspace(float(lo), float(hi), int(n))


def _compute_ele(spec):
    p = _p(spec)
    cfg = load_analysis_config(_rel(spec["_path"]))
    sd = cfg.signal                                          # EleBeamSignalDef
    ado_cache, ref_cache = {}, {}

    def ado_for(bank):
        if bank not in ado_cache:
            ado_cache[bank] = SG.ele_signal(_rel(bank), sd)
        return ado_cache[bank]

    def ref_for(refs):
        keyt = tuple(refs)
        if keyt not in ref_cache:
            parts = []
            for i, pth in enumerate(refs):
                r = SG.ele_oracle_signal(_rel(pth), sd)
                r["chan"] = np.full(len(r["w"]), i, int)     # file order: qe -> 0 (QE), res -> 1 (RES)
                parts.append(r)
            ref_cache[keyt] = {k: np.concatenate([r[k] for r in parts]) for k in parts[0]}
        return ref_cache[keyt]

    default_bank = (cfg.inputs.get("adonis_bank") or [None])[0]
    default_ref = cfg.inputs.get("reference") or []
    panels, ann, rxmax = [], {}, {}
    for i, ps in enumerate(p["panels"]):
        ado = ado_for(ps.get("bank") or default_bank)
        ref = ref_for(ps.get("ref") or default_ref)
        am = _TOPO[ps.get("topo", "incl")](ado); rm = _TOPO[ps.get("topo", "incl")](ref)
        sc = float(ps.get("value_scale", 1.0))               # e.g. MeV -> GeV on the plotted omega
        key = f'{ps["obs"]}_{i}'
        panels.append({"key": key, "edges": _edges_of(ps), "label": ps["label"],
                       "ado": (ado[ps["obs"]][am] * sc, ado["w"][am], ado["chan"][am]),
                       "ref": (ref[ps["obs"]][rm] * sc, ref["w"][rm], ref["chan"][rm])})
        if ps.get("annotate"):
            ann[key] = ps["annotate"]
        if ps.get("ratio_xmax") is not None:                 # stop the ratio at a kinematic cliff (E_cal)
            rxmax[key] = float(ps["ratio_xmax"])
    specs, ado_sel, ref_sel = _assemble(panels)
    _suppress_breakdown(p, ado_sel, ref_sel)                  # breakdown: false -> single IBM-blue line (fig0456)
    layout = dict(title=cfg.title, ratio_ylim=tuple(cfg.ratio_ylim), annotations=ann, ylabel=p.get("ylabel"),
                  ratio_xmax=rxmax or None, panel_kw=_panel_kw(p))
    return specs, ado_sel, ref_sel, layout


_COMPUTE = {"sliced": _compute_sliced, "ele": _compute_ele}


# =================================================================================== RENDER: multiobs
def render_multiobs(spec):
    """Pure selection-histogram figure: straight through run_analysis under the paper style."""
    cfg = load_analysis_config(_rel(spec["_path"]))
    return run_analysis(cfg, **style.paper_run_analysis_kw())


# =================================================================================== RENDER: panels
def render_panels(spec):
    """Compute a panel set (compute selected by spec['compute']) and draw it with make_figure -- the
    shared grid-of-chi2_ratio_panel builder.  Layout kwargs come from the compute + spec['layout']."""
    specs, ado_sel, ref_sel, layout = _COMPUTE[spec["compute"]](spec)
    layout.update(spec.get("layout", {}) or {})
    kw = dict(ado_label="ADoNIS", ref_label="ACHILLES", panel_w=2.4, fig_h=2.6, min_w=3.2,
              legend_fn=style.panel_legend, label_as_xlabel=True,
              panel_kw=layout.pop("panel_kw", style.panel_kw(ratio_yticks=[0.8, 1.0, 1.2])),
              # rect_top=1.0 packs the panels flush; the suptitle sits just above them (y=0.95).  The
              # old rect_top=0.95 / y=0.995 left a large title gap under make_figure's nested gridspec.
              title_kw={"fontsize": 9, "y": 0.95, "va": "top"}, rect_top=1.0)
    kw.update(layout)
    out = ROOT / f"output/paper/{spec['name']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, results, sig = make_figure(specs, ref_sel, ado_sel, **kw)
    fig.savefig(out, dpi=300, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    for k, r in results.items():
        print(f"  {k}: chi2/ndf = {r['chi2']/max(r['ndf'],1):.2f}  ACH/ADO = {r['ach_ado']:.4f}")
    return {"results": results, "sigma": sig}


# =================================================================================== RENDER: bespoke
# figs 2, 3, 13 are multi-series overlays / angular samplers -- NOT one-observable panels, so they draw
# with matplotlib directly (still here, still one entry point).  save via style.save (paper .png+.pdf).
def render_sigma_channels(spec):
    """Fig 2 -- free-nucleon RES sigma(E_nu), 3 CC channels overlaid in one panel + a 3-channel ratio."""
    import matplotlib.pyplot as plt
    from analysis.paper.freenucleon_bank import ENERGIES, CHANNEL_SPECS, load_scan
    NB = 1.0e5                                                     # nb -> 10^-38 cm^2 (plotted axis unit)
    SCAN = _rel((spec.get("inputs") or {}).get("oracle_scan", "output/oracle_freenucleon_scan"))

    def ach_sigma(E, sp, pi_pid):
        fs = sorted(glob.glob(f"{SCAN}/{sp}_E{int(E)}/*.npz"))
        if not fs:
            return np.nan, np.nan
        d = np.load(fs[0], allow_pickle=True); w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
        m = np.asarray(d["pi_pid"]) == pi_pid
        return float(w[m].sum()), float(np.sqrt((w[m] ** 2).sum()))

    style.use(); scan = load_scan()
    fig, ax = plt.subplots(2, 1, figsize=(4.6, 3.4), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    top, bot = ax; COLS = (style.C_TOTAL, style.C_QE, style.C_RES)
    for c, (tag, _ci, tex, sp, ach_pi) in enumerate(CHANNEL_SPECS):
        a_col, h_col = style.lighter(COLS[c]), style.darker(COLS[c])
        aS, aE = (v * NB for v in scan[tag])
        hS, hE = (np.array(v) * NB for v in zip(*(ach_sigma(E, sp, ach_pi) for E in ENERGIES)))
        top.fill_between(ENERGIES, aS - aE, aS + aE, color=a_col, alpha=0.25, lw=0)
        top.plot(ENERGIES, aS, "-", color=a_col, lw=1.6)
        top.errorbar(ENERGIES, hS, yerr=hE, fmt="s", ms=3.0, color=h_col, capsize=1.5, lw=1.0, ls="--", zorder=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = aS / hS; re = np.abs(r) * np.sqrt((aE / aS) ** 2 + (hE / np.where(hS > 0, hS, np.nan)) ** 2)
        bot.errorbar(ENERGIES, r, yerr=re, fmt="o", ms=2.5, color=COLS[c], capsize=1.5, lw=1.0)
    top.set_xlim(0, 4600); top.set_ylim(bottom=0)
    top.set_ylim(top.get_ylim()[0], top.get_ylim()[1] * 1.30); top.set_ylabel(r"$\sigma$ [$10^{-38}$ cm$^2$]")
    h, l, hm = style.swatches([(tex, COLS[c]) for c, (_t, _ci, tex, _s, _p) in enumerate(CHANNEL_SPECS)])
    top.legend(h, l, handler_map=hm, loc="upper left", fontsize=7, handlelength=3.0, labelspacing=0.3, borderpad=0.2)
    bot.axhline(1.0, ls="-", color="0.6", lw=0.8)
    for off in (0.1, 0.2):
        bot.axhline(1.0 - off, ls="--", color="0.7", lw=0.6); bot.axhline(1.0 + off, ls="--", color="0.7", lw=0.6)
    bot.set_ylim(0.6, 1.4); bot.set_yticks([0.8, 1.0, 1.2])
    bot.set_xlabel(r"$E_\nu$ [MeV]"); bot.set_ylabel("ratio")
    fig.suptitle(r"Free-nucleon RES single-pion $\sigma(E_\nu)$", fontsize=9, y=0.995, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); style.save(fig, spec["name"])


def render_beam_sigma(spec):
    """Fig 3 -- pi+ nucleus absorption+reaction sigma(p), 2x2 curve blocks (nucleus rows, channel cols)."""
    import os
    import matplotlib.pyplot as plt
    pp = _p(spec)
    BANK = pp.get("bank_pattern", "output/paper_banks_p4/beam_{beam}_{target}/merged")
    os.environ.setdefault("ADONIS_BEAM_PATTERN", BANK)
    from analysis.paper.beams.make_figs import adonis_sigma
    from analysis.paper.beams import achilles_beam as AB
    nbins = int(pp.get("nbins", 30)); BEAM = pp.get("beam", "pip")

    def reduce_nuc(nuc):
        def _b():
            edges, sr, ss, er, es = adonis_sigma(BEAM, nbins, nuc)
            hr, hs, her, hes, _nr, _ns, _nt = AB.sigma_of_p(BEAM, edges, nuc)
            return dict(edges=edges, sr=sr, ss=ss, er=er, es=es, hr=hr, hs=hs, her=her, hes=hes)
        d = plotcache.cached(f"fig03_{BEAM}_{nuc}_n{nbins}", _b,
                             deps=[ROOT / BANK.format(beam=BEAM, target=nuc), *AB.source_paths(BEAM, nuc)],
                             params={"beam": BEAM, "target": nuc, "nbins": nbins})
        return d["edges"], d["sr"], d["ss"], d["er"], d["es"], d["hr"], d["hs"], d["her"], d["hes"]

    style.use(); fig = plt.figure(figsize=(7.1, 4.6))
    outer = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28); ax = {}
    for ni in range(2):
        for oi in range(2):
            inner = outer[ni, oi].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
            a0 = fig.add_subplot(inner[0]); ax[ni, oi] = (a0, fig.add_subplot(inner[1], sharex=a0))
            a0.tick_params(labelbottom=False)
    for ni, nuc in enumerate(("C", "Ar")):
        edges, sr, ss, er, es, hr, hs, her, hes = reduce_nuc(nuc)
        cen = 0.5 * (edges[:-1] + edges[1:]) / 1000.0; PMAX = {"absorption": 0.5, "reaction": 1.0}
        for oi, (A, EA, H, EH, lab) in enumerate(((ss, es, hs, hes, "absorption"), (sr, er, hr, her, "reaction"))):
            a0, a1 = ax[ni, oi]; k = cen <= PMAX[lab]; edg = edges[:int(k.sum()) + 1] / 1000.0
            res = chi2_ratio_panel(a0, a1, None, {"x": cen[k], "y": H[k], "yerr": EH[k], "edges": edg},
                                   {"x": cen[k], "y": A[k], "yerr": EA[k]}, label="",
                                   xlabel=r"beam $|p|$ [GeV/c]", ratio_ylim=(0.8, 1.2),
                                   ado_label="ADoNIS", ref_label="ACHILLES", total_color=style.C_QE,
                                   ratio_color=style.C_RATIO, ado_lighten=style.ADO_LIGHTEN,
                                   ref_darken=style.REF_DARKEN, headroom=0.30)
            a0.set_ylabel(r"$\sigma$ [mb]"); a1.set_ylabel("ratio"); a1.set_yticks([0.9, 1.0, 1.1])
            if ni == 0:
                a0.set_title(lab, fontsize=9); a1.set_xlabel("")
            a0.text(0.97, 0.95, rf"$\pi^+$ {_NUC_TEX[nuc]}", transform=a0.transAxes, fontsize=8, va="top", ha="right")
    fig.suptitle(r"$\pi^+$ absorption + reaction on $^{12}$C / $^{40}$Ar (pure transport)", fontsize=9, y=0.985)
    style.save(fig, spec["name"])


def render_dcc(spec):
    """Fig 13 -- meson-baryon DCC: total sigma(W) for 4 species (analytic + INC MC) + pi+p angular sampler."""
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    from adonis.fsi.interactions.meson_baryon_amplitudes import (
        load_anl, _channel_sigma, pim_p_total, dsigma_dOmega,
        conversion_sigma_grid, eta_elastic_sigma_grid, eta_backconv_sigma_grid)
    from adonis.fsi.interactions.meson_baryon_xsec import jax_sample_cos_cm
    from adonis.constants import mpip as M_PI, mN as M_N   # canonical masses (no local roundings)
    _R2 = np.sqrt(2.0) / 3.0; MB_FM2 = 0.1                 # isospin C-G factor; mb<->fm^2 unit (not masses)

    def _interp(Wg, Ws, ys):
        return np.interp(Wg, Ws, ys, left=0.0, right=0.0)

    def totals():
        Wg, amps = load_anl(0, 0)
        piN_pip = _channel_sigma(amps, Wg, {3: 1.0})
        piN_pi0 = (_channel_sigma(amps, Wg, {3: 2.0 / 3, 1: 1.0 / 3}) + _channel_sigma(amps, Wg, {3: _R2, 1: -_R2}))
        piN_pim = pim_p_total(Wg)[1]
        Wc, conv = conversion_sigma_grid(); We, eel = eta_elastic_sigma_grid(); Wb, ebk = eta_backconv_sigma_grid()
        eta = _interp(Wg, We, eel) + _interp(Wg, Wb, ebk[0].sum(axis=0))
        return Wg, {r"$\pi^+ p$": ("tab:red", piN_pip + _interp(Wg, Wc, conv[0, 0])),
                    r"$\pi^0 p$": ("tab:blue", piN_pi0 + _interp(Wg, Wc, conv[1, 0])),
                    r"$\pi^- p$": ("tab:green", piN_pim + _interp(Wg, Wc, conv[2, 0])),
                    r"$\eta p$": ("tab:purple", eta)}

    def inc_mc(sig_mb, rng, ntrial=4000, k=12.0):
        s = np.maximum(sig_mb * MB_FM2, 1e-9); R2 = k * s / np.pi
        b2 = rng.random((len(s), ntrial)) * R2[:, None]; P = np.exp(-np.pi * b2 / s[:, None])
        hit = rng.random(P.shape) < P; ph = hit.mean(1)
        return (np.pi * R2 * ph) / MB_FM2, (np.pi * R2 * np.sqrt(np.clip(ph * (1 - ph), 0, None) / ntrial)) / MB_FM2

    style.use(); rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.7))
    W, curves = totals(); WG = W / 1000.0; Wc = W[::3]
    for lab, (col, sig) in curves.items():
        ax[0].plot(WG, sig, "-", color=col, lw=1.6, label=lab, zorder=2)
        mc, err = inc_mc(np.interp(Wc, W, sig), rng)
        ax[0].errorbar(Wc / 1000.0, mc, yerr=err, fmt="o", ms=2.6, color=col, mfc="white",
                       elinewidth=0.7, capsize=0, lw=0, zorder=3)
    ax[0].plot([], [], "-", color="0.4", label="ANL-Osaka (analytic)")
    ax[0].plot([], [], "o", color="0.4", mfc="white", label="ADoNIS INC (MC)")
    ax[0].set_xlim(1.08, 2.0); ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r"$W$ [GeV]"); ax[0].set_ylabel(r"$\sigma$ [mb]")
    ax[0].legend(fontsize=8, ncol=2, title="off proton", title_fontsize=8)
    ax[0].set_title(r"total meson-baryon $\sigma(W)$", fontsize=10)
    W300 = float(np.sqrt(M_PI ** 2 + M_N ** 2 + 2 * M_N * np.sqrt(300.0 ** 2 + M_PI ** 2)))
    cg = np.linspace(-1, 1, 200); dd = dsigma_dOmega(W300, cg, {3: 1.0}, i=0, f=0)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    ax[1].plot(cg, dd / (_trap(dd, cg) * 2 * np.pi), "-", color="tab:red", lw=1.8,
               label=rf"ANL-Osaka  ($W$={W300/1000:.2f} GeV)")
    N = 200_000; u = jax.random.uniform(jax.random.PRNGKey(1), (N,))
    cs = np.asarray(jax_sample_cos_cm(jnp.full((N,), W300), u, chan=0))
    edges = np.linspace(-1, 1, 21); ctr = 0.5 * (edges[1:] + edges[:-1])
    cnt, _ = np.histogram(cs, edges); bw = np.diff(edges)
    dens = cnt / (N * bw * 2 * np.pi); derr = np.sqrt(cnt) / (N * bw * 2 * np.pi)
    ax[1].errorbar(ctr, dens, yerr=derr, fmt="o", ms=3.5, color="tab:red", mfc="white",
                   elinewidth=0.8, capsize=0, lw=0, label=r"ADoNIS sampler ($\pi^+ p$)")
    ax[1].set_xlim(-1, 1); ax[1].set_ylim(0, None)
    ax[1].set_xlabel(r"$\cos(\theta_{\rm CM})$"); ax[1].set_ylabel(r"$(1/\sigma)\, d\sigma/d\Omega$")
    ax[1].legend(fontsize=8); ax[1].set_title(r"$\pi^+ p$ angular at $p=300$ MeV", fontsize=10)
    fig.suptitle(r"ADoNIS vs ANL-Osaka --- meson-baryon DCC (shared cascade cross sections)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); style.save(fig, spec["name"])


RENDERERS = {"multiobs": render_multiobs, "panels": render_panels,
             "sigma_channels": render_sigma_channels, "beam_sigma": render_beam_sigma, "dcc": render_dcc}


def render(spec):
    """Dispatch one figure spec (dict, with '_path' injected by make.py) to its render function."""
    return RENDERERS[spec["render"]](spec)
