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
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from adonis.workflow.analyze import run_analysis                 # noqa: E402
from adonis.workflow.config import load_analysis_config, SignalDef   # noqa: E402
from adonis.workflow import selection as SG                      # noqa: E402
from adonis.workflow.plotting import make_figure, chi2_ratio_panel   # noqa: E402
from analysis.paper import style                                 # noqa: E402
from analysis.paper import plotcache                             # noqa: E402


def _p(spec):
    return dict((spec or {}).get("params", {}) or {})


def _rel(path):
    return str(ROOT / path)


# =============================================================== COMPUTE: sliced selection (figs 10, 11)
def _compute_sliced(spec):
    """Selection histogrammed in SLICES of a second variable -> (specs, ado_sel, ref_sel, layout).
    Config: params.slice_by/slice_edges (the slice variable + its edges), params.obs/obs_edges (the
    histogrammed observable + per-slice edges, or one edge set), params.signal_nc (NC selection)."""
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

    specs, ma, mr = [], {}, {}
    sv_a = np.asarray(ado[slice_by], float) * slice_scale
    sv_r = np.asarray(ref[slice_by], float) * slice_scale
    parts_a = {"w": [], "chan": []}; parts_r = {"w": [], "chan": []}
    for i in range(nslice):
        lo, hi = s_edges[i], s_edges[i + 1]
        key = f"{obs}_s{i}"
        labels = p.get("labels")
        label = labels[i] if labels else rf"{p.get('obs_label', obs)},  ${lo:g}<{stex}<{hi:g}$"
        specs.append((key, np.asarray(oe[i], float) * edge_scale, label))
        ka = (sv_a >= lo) & (sv_a < hi); kr = (sv_r >= lo) & (sv_r < hi)
        ma[key] = (np.asarray(ado[obs], float), ka)
        mr[key] = (np.asarray(ref[obs], float), kr)
        parts_a["w"].append(np.asarray(ado["w"], float)[ka]); parts_a["chan"].append(np.asarray(ado["chan"])[ka])
        parts_r["w"].append(np.asarray(ref["w"], float)[kr]); parts_r["chan"].append(np.asarray(ref["chan"])[kr])

    # make_figure wants one dict: concatenate the per-slice weights, and give each key NaN outside its slice
    ado_sel = {"w": np.concatenate(parts_a["w"]), "chan": np.concatenate(parts_a["chan"])}
    ref_sel = {"w": np.concatenate(parts_r["w"]), "chan": np.concatenate(parts_r["chan"])}
    off_a = np.cumsum([0] + [len(w) for w in parts_a["w"]])
    off_r = np.cumsum([0] + [len(w) for w in parts_r["w"]])
    for i, (key, _e, _l) in enumerate(specs):
        va = np.full(len(ado_sel["w"]), np.nan); va[off_a[i]:off_a[i + 1]] = ma[key][0][ma[key][1]]
        vr = np.full(len(ref_sel["w"]), np.nan); vr[off_r[i]:off_r[i + 1]] = mr[key][0][mr[key][1]]
        ado_sel[key] = va; ref_sel[key] = vr
    layout = dict(title=cfg.title, ratio_band=tuple(cfg.ratio_band), ratio_ylim=tuple(cfg.ratio_ylim))
    return specs, ado_sel, ref_sel, layout


# =============================================================== COMPUTE: (e,e') omega, QE/RES (fig 1)
_EB1 = 2222.0
_THE = (14.0, 17.0)
_NUC_TEX = {"C": r"$^{12}$C", "Ar": r"$^{40}$Ar"}


def _ee_adonis(nuc):
    bd = ROOT / f"output/paper_banks_p4/beam_e_{nuc}/merged"
    nch = json.load(open(bd / "manifest.json"))["n_chunks"]
    O, W, CH = [], [], []
    for f in sorted(glob.glob(str(bd / "chunk_*.npz"))):
        d = np.load(f); th = np.asarray(d["theta"], float); k = (th >= _THE[0]) & (th <= _THE[1])
        O.append(np.asarray(d["omega"], float)[k] / 1000.0)
        W.append(np.asarray(d["c"], float)[k] / nch)
        CH.append(np.asarray(d["channel"])[k])
    return np.concatenate(O), np.concatenate(W), np.concatenate(CH)


def _ee_achilles(nuc, ch):
    d = np.load(ROOT / f"output/achilles/fsrich/inclusive_ee_{nuc}_{ch}.npz", allow_pickle=True)
    lep = np.asarray(d["lep"], float); Ee = lep[:, 0]
    pe = np.linalg.norm(lep[:, 1:], axis=1)
    cth = np.where(pe > 0, lep[:, 3] / np.maximum(pe, 1e-9), -2.0)
    the = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    k = (the >= _THE[0]) & (the <= _THE[1])
    w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
    return (_EB1 - Ee)[k] / 1000.0, w[k]


def _compute_ee_domega(spec):
    edges = np.linspace(0.05, 0.95, 46)
    specs, ado_sel, ref_sel = [], {"w": [], "chan": []}, {"w": [], "chan": []}
    off_a, off_r = [0], [0]
    ao_all, hv_all = {}, {}
    for nuc in ("Ar", "C"):                                       # paper order: Ar left, C right
        ao, aw, ach = _ee_adonis(nuc)
        hqo, hqw = _ee_achilles(nuc, "qe"); hro, hrw = _ee_achilles(nuc, "res")
        ho = np.concatenate([hqo, hro]); hw = np.concatenate([hqw, hrw])
        hch = np.concatenate([np.zeros(len(hqo), int), np.ones(len(hro), int)])
        key = f"omega_{nuc}"
        specs.append((key, edges, r"$\omega$ [GeV]"))
        ado_sel["w"].append(aw); ado_sel["chan"].append(ach); ao_all[key] = ao
        ref_sel["w"].append(hw); ref_sel["chan"].append(hch); hv_all[key] = ho
        off_a.append(off_a[-1] + len(aw)); off_r.append(off_r[-1] + len(hw))
    ado_sel["w"] = np.concatenate(ado_sel["w"]); ado_sel["chan"] = np.concatenate(ado_sel["chan"])
    ref_sel["w"] = np.concatenate(ref_sel["w"]); ref_sel["chan"] = np.concatenate(ref_sel["chan"])
    for i, (key, _e, _l) in enumerate(specs):
        va = np.full(len(ado_sel["w"]), np.nan); va[off_a[i]:off_a[i + 1]] = ao_all[key]
        vr = np.full(len(ref_sel["w"]), np.nan); vr[off_r[i]:off_r[i + 1]] = hv_all[key]
        ado_sel[key] = va; ref_sel[key] = vr
    ann = {f"omega_{n}": _NUC_TEX[n] for n in ("Ar", "C")}
    layout = dict(title=r"Inclusive (e,e') at 2.222 GeV, $\theta_{e'}\approx15.5^\circ$",
                  ratio_ylim=(0.6, 1.4), annotations=ann,
                  ylabel=r"$d\sigma/d\omega$ [nb/GeV]")
    return specs, ado_sel, ref_sel, layout


# =============================================================== COMPUTE: e4nu E_QE/E_cal/P_T (figs 4-6)
_EB4 = 1159.0; _MNUC = 938.9; _MP = 938.272; _ME = 0.511; _EPS = 21.0
_PP_MIN = 300.0; _TP = (10.0, 140.0); _EE_MIN = 400.0; _THE4 = (15.0, 45.0)
_BANK4 = str(ROOT / "output/paper_banks_p4/beam_e_C_1159/merged")
_ORA4 = [str(ROOT / f"output/achilles/fsrich/ee_C_1159_{c}_fsi.npz") for c in ("qe", "res")]


def _e4nu_eqe(Ee, cth):
    pe = np.sqrt(np.maximum(Ee ** 2 - _ME ** 2, 0.0))
    return (2 * _MNUC * _EPS + 2 * _MNUC * Ee - _ME ** 2) / (2 * (_MNUC - Ee + pe * cth))


def _e4nu_lead(pid, p4, seg, n):
    mom = np.linalg.norm(p4[:, 1:], axis=1)
    cth = np.where(mom > 0, p4[:, 3] / np.maximum(mom, 1e-9), -2.0)
    th = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    acc = (pid == 2212) & (mom > _PP_MIN) & (th >= _TP[0]) & (th <= _TP[1])
    nprot = np.zeros(n); np.add.at(nprot, seg[acc], 1.0)
    key = np.where(acc, mom, -1.0); mx = np.full(n, -1.0); np.maximum.at(mx, seg, key)
    lead = np.zeros((n, 4)); islead = acc & (key == mx[seg]) & (mom > 0)
    lead[seg[islead]] = p4[islead]
    return lead, nprot.astype(int)


def _e4nu_adonis():
    EQ, wq, EC, PT, wc = [], [], [], [], []
    man = json.load(open(Path(_BANK4) / "manifest.json")); nch = man["n_chunks"]
    for f in sorted(glob.glob(_BANK4 + "/chunk_*.npz")):
        d = np.load(f)
        c = np.asarray(d["c"], float) / nch
        om = np.asarray(d["omega"], float); th = np.radians(np.asarray(d["theta"], float))
        Ee = _EB4 - om; cth = np.cos(th)
        elec = (Ee >= _EE_MIN) & (np.degrees(th) >= _THE4[0]) & (np.degrees(th) <= _THE4[1])
        npi = np.asarray(d["n_pi_out"])
        k0 = elec & (npi == 0); EQ.append(_e4nu_eqe(Ee, cth)[k0]); wq.append(c[k0])
        ne = len(c); seg = np.repeat(np.arange(ne), np.diff(np.asarray(d["fs_off"], np.int64)))
        lead, nprot = _e4nu_lead(np.asarray(d["fs_pid"]), np.asarray(d["fs_p4"], float), seg, ne)
        k1 = elec & (npi == 0) & (nprot == 1); Tp = lead[:, 0] - _MP
        EC.append((Ee + Tp + _EPS)[k1]); wc.append(c[k1])
        ke = np.asarray(d["k_lep"] if "k_lep" in d.files else d["k_e"], float)
        pt = np.sqrt((ke[:, 1] + lead[:, 1]) ** 2 + (ke[:, 2] + lead[:, 2]) ** 2)
        PT.append(pt[k1])
    return (np.concatenate(EQ), np.concatenate(wq)), (np.concatenate(EC), np.concatenate(wc), np.concatenate(PT))


def _e4nu_achilles():
    EQ, wq, EC, PT, wc = [], [], [], [], []
    for p in _ORA4:
        d = np.load(p, allow_pickle=True)
        w = np.asarray(d["w"], float) * float(d["weight_to_nb"])
        lep = np.asarray(d["lep"], float); Ee = lep[:, 0]
        pe = np.linalg.norm(lep[:, 1:], axis=1); cth = np.where(pe > 0, lep[:, 3] / np.maximum(pe, 1e-9), -2.0)
        the = np.degrees(np.arccos(np.clip(cth, -1, 1)))
        elec = (Ee >= _EE_MIN) & (the >= _THE4[0]) & (the <= _THE4[1])
        npi = (np.asarray(d["pi_pid"]) != 0).sum(1)
        k0 = elec & (npi == 0); EQ.append(_e4nu_eqe(Ee, cth)[k0]); wq.append(w[k0])
        prot = np.asarray(d["prot_p4"], float); pm = np.linalg.norm(prot[:, :, 1:], axis=2)
        pcz = np.where(pm > 0, prot[:, :, 3] / np.maximum(pm, 1e-9), -2.0)
        tp = np.degrees(np.arccos(np.clip(pcz, -1, 1)))
        acc = (pm > _PP_MIN) & (tp >= _TP[0]) & (tp <= _TP[1]); nprot = acc.sum(1)
        key = np.where(acc, pm, -1.0); j = key.argmax(1); lead = prot[np.arange(len(lep)), j]
        k1 = elec & (npi == 0) & (nprot == 1); Tp = lead[:, 0] - _MP
        EC.append((Ee + Tp + _EPS)[k1]); wc.append(w[k1])
        pt = np.sqrt((lep[:, 1] + lead[:, 1]) ** 2 + (lep[:, 2] + lead[:, 2]) ** 2); PT.append(pt[k1])
    return (np.concatenate(EQ), np.concatenate(wq)), (np.concatenate(EC), np.concatenate(wc), np.concatenate(PT))


def _compute_e4nu(spec):
    def _b():
        (aEQ, awq), (aEC, awc, aPT) = _e4nu_adonis()
        (hEQ, hwq), (hEC, hwc, hPT) = _e4nu_achilles()
        return dict(aEQ=aEQ, awq=awq, aEC=aEC, awc=awc, aPT=aPT, hEQ=hEQ, hwq=hwq, hEC=hEC, hwc=hwc, hPT=hPT)
    d = plotcache.cached("fig456_e4nu", _b, deps=[_BANK4, *_ORA4],
                         params={"eb": _EB4, "pp_min": _PP_MIN, "tp": _TP, "ee_min": _EE_MIN,
                                 "the": _THE4, "eps": _EPS})
    specs = [("E_QE", np.linspace(600, 1300, 25), r"$E_{QE}$ [MeV]"),
             ("E_cal", np.linspace(700, 1300, 25), r"$E_{cal}$ [MeV]"),
             ("P_T", np.linspace(0, 600, 25), r"$P_T$ [MeV/c]")]
    ado_sel = {"E_QE": d["aEQ"], "E_cal": d["aEC"], "P_T": d["aPT"]}
    ref_sel = {"E_QE": d["hEQ"], "E_cal": d["hEC"], "P_T": d["hPT"]}
    # each panel has its own weights (0pi for E_QE, 1p0pi for E_cal/P_T) -> pass per-panel via NaN pad
    # (make_figure uses one 'w'); here the 3 panels share the weight vectors by selection, so build a
    # combined weight and NaN-pad each observable, mirroring the sliced path.
    w_ado = {"E_QE": d["awq"], "E_cal": d["awc"], "P_T": d["awc"]}
    w_ref = {"E_QE": d["hwq"], "E_cal": d["hwc"], "P_T": d["hwc"]}
    off_a = np.cumsum([0] + [len(w_ado[k]) for k, _e, _l in specs])
    off_r = np.cumsum([0] + [len(w_ref[k]) for k, _e, _l in specs])
    AW = {"w": np.concatenate([w_ado[k] for k, _e, _l in specs])}
    RW = {"w": np.concatenate([w_ref[k] for k, _e, _l in specs])}
    for i, (k, _e, _l) in enumerate(specs):
        va = np.full(len(AW["w"]), np.nan); va[off_a[i]:off_a[i + 1]] = ado_sel[k]; AW[k] = va
        vr = np.full(len(RW["w"]), np.nan); vr[off_r[i]:off_r[i + 1]] = ref_sel[k]; RW[k] = vr
    ann = {"E_QE": r"0$\pi$", "E_cal": r"1p0$\pi$", "P_T": r"1p0$\pi$"}
    layout = dict(title=r"e4$\nu$ (e,e') on $^{12}$C at 1.159 GeV", ratio_ylim=(0.6, 1.4),
                  annotations=ann, ylabel=r"$d\sigma/dx$ [nb]",
                  panel_kw=style.panel_kw(ratio_yticks=None))
    return specs, AW, RW, layout


_COMPUTE = {"sliced": _compute_sliced, "ee_domega": _compute_ee_domega, "e4nu": _compute_e4nu}


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
              title_kw={"fontsize": 9, "y": 0.995, "va": "top"}, rect_top=0.95)
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
    NB = 1.0e5; SCAN = _rel("output/oracle_freenucleon_scan")

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
    os.environ.setdefault("ADONIS_BEAM_PATTERN", "output/paper_banks_p4/beam_{beam}_{target}/merged")
    from analysis.paper.beams.make_figs import adonis_sigma
    from analysis.paper.beams import achilles_beam as AB
    nbins = int(_p(spec).get("nbins", 30)); BEAM = "pip"
    BANK = "output/paper_banks_p4/beam_{beam}_{target}/merged"

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
    _R2 = np.sqrt(2.0) / 3.0; M_PI = 139.57018; M_N = 938.918754; MB_FM2 = 0.1

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
