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
from adonis import cache as plotcache                             # noqa: E402


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
    layout = dict(ratio_band=tuple(cfg.ratio_band), ratio_ylim=tuple(cfg.ratio_ylim),
                  panel_kw=_panel_kw(p))   # no title -> caption carries it
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
    layout = dict(ratio_ylim=tuple(cfg.ratio_ylim), annotations=ann, ylabel=p.get("ylabel"),
                  ratio_xmax=rxmax or None, panel_kw=_panel_kw(p))   # no title -> caption carries it
    return specs, ado_sel, ref_sel, layout


_COMPUTE = {"sliced": _compute_sliced, "ele": _compute_ele}


# =================================================================================== RENDER: multiobs
def render_multiobs(spec, show_ratio=True):
    """Pure selection-histogram figure: straight through run_analysis under the paper style.
    show_ratio=False drops the ACH/ADO ratio strip and writes to a *_noratio file (paper fig untouched)."""
    cfg = load_analysis_config(_rel(spec["_path"]))
    if not show_ratio:
        cfg.out_path = cfg.out_path.replace(".png", "_noratio.png")
    return run_analysis(cfg, title="", show_ratio=show_ratio, **style.paper_run_analysis_kw())   # title -> caption


# =================================================================================== RENDER: panels
def render_panels(spec, show_ratio=True):
    """Compute a panel set (compute selected by spec['compute']) and draw it with make_figure -- the
    shared grid-of-chi2_ratio_panel builder.  Layout kwargs come from the compute + spec['layout'].
    show_ratio=False drops the ACH/ADO ratio strip and writes to a *_noratio file (paper fig untouched)."""
    specs, ado_sel, ref_sel, layout = _COMPUTE[spec["compute"]](spec)
    layout.update(spec.get("layout", {}) or {})
    kw = dict(ado_label="ADoNIS", ref_label="ACHILLES",
              panel_w=style.PANEL_W, fig_h=style.PANEL_H, min_w=style.PANEL_W, max_cols=style.STD_COLS,
              legend_fn=style.panel_legend, label_as_xlabel=True,
              panel_kw=layout.pop("panel_kw", style.panel_kw(ratio_yticks=[0.8, 1.0, 1.2])),
              rect_top=1.0)   # no title on the plot (the caption carries it) -> pack flush to the top
    kw.update(layout)
    kw["show_ratio"] = show_ratio
    out = ROOT / f"output/paper/{spec['name']}{'' if show_ratio else '_noratio'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, results, sig = make_figure(specs, ref_sel, ado_sel, **kw)
    fig.savefig(out, dpi=300, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")
    for k, r in results.items():
        print(f"  {k}: chi2/ndf = {r['chi2']/max(r['ndf'],1):.2f}  ACH/ADO = {r['ach_ado']:.4f}")
    return {"results": results, "sigma": sig}


def render_beam_sigma(spec, show_ratio=True):
    """Fig 3 -- pi+ nucleus absorption+reaction sigma(p), 2x2 curve blocks (nucleus rows, channel cols).
    show_ratio=False drops the per-block ACH/ADO ratio strip and writes to a *_noratio file."""
    import os
    import matplotlib.pyplot as plt
    pp = _p(spec)
    BANK = pp.get("bank_pattern", "output/paper_banks_p4/beam_{beam}_{target}/merged")
    os.environ.setdefault("ADONIS_BEAM_PATTERN", BANK)
    from analysis.campaign.beams import bank_sigma as adonis_sigma
    from analysis.oracle_tools import beam_sigma as AB
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

    # 2x2 of (top+ratio) blocks at the shared panel proportions: 2 cols x PANEL_W, 2 rows x PANEL_H.
    # no ratio -> each block is a single axis; shrink the row height to the top panel's 3/4 share.
    style.use(); fig = plt.figure(figsize=(2 * style.PANEL_W, 2 * style.PANEL_H * (1.0 if show_ratio else 0.75)))
    outer = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.28); ax = {}   # tight row gap (matches make_figure)
    for ni in range(2):
        for oi in range(2):
            if show_ratio:
                inner = outer[ni, oi].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
                a0 = fig.add_subplot(inner[0]); ax[ni, oi] = (a0, fig.add_subplot(inner[1], sharex=a0))
                a0.tick_params(labelbottom=False)
            else:
                ax[ni, oi] = (fig.add_subplot(outer[ni, oi]), None)
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
                                   ref_darken=style.REF_DARKEN, headroom=0.30, show_ratio=show_ratio)
            a0.set_ylabel(r"$\sigma$ [mb]")
            if show_ratio:
                a1.set_ylabel("ratio"); a1.set_yticks([0.9, 1.0, 1.1])
            if ni == 0:
                a0.set_title(lab, fontsize=9); (a1 if show_ratio else a0).set_xlabel("")
            a0.text(0.97, 0.95, rf"$\pi^+$ {_NUC_TEX[nuc]}", transform=a0.transAxes, fontsize=8, va="top", ha="right")
    style.save(fig, spec["name"] + ("" if show_ratio else "_noratio"))   # no title -> caption carries it


# Only the three renderers the paper's specs actually select.  render_sigma_channels (fig 2) and
# render_dcc (fig 13) were removed with their figures: no spec named them, so they were reachable
# only by editing this table.
RENDERERS = {"multiobs": render_multiobs, "panels": render_panels,
             "beam_sigma": render_beam_sigma}


def render(spec, show_ratio=True):
    """Dispatch one figure spec (dict, with '_path' injected by make.py) to its render function.
    show_ratio=False renders the ratio-less variant to a *_noratio file (see make.py --no-ratio)."""
    return RENDERERS[spec["render"]](spec, show_ratio=show_ratio)


def render_sample(cfg_path, show_ratio=True):
    """Render the sec1 ADoNIS-vs-ACHILLES figure for a sample config.

    Was AnaSample.plot().  It lived in adonis/analysis/sample.py and imported this module, which made
    the core depend on one section's figure code; the sample config IS a sec1 spec, so the verb belongs
    on this side of the boundary.  render_multiobs/render_panels re-load the AnalysisConfig from _path,
    so one config file still drives both this and AnaSample.gate1().
    """
    import yaml
    from pathlib import Path
    spec = yaml.safe_load(Path(cfg_path).read_text()) or {}
    spec["_path"] = str(cfg_path); spec.setdefault("name", Path(cfg_path).stem)
    return render(spec, show_ratio=show_ratio)
