"""run_analysis(AnalysisConfig): the config-driven ADoNIS-vs-ACHILLES figure driver + CLI.

    python -m adonis.workflow.analyze <fig.yaml>     ->  output/.../<name>.pdf + .png

ADoNIS  = selection.bank_signal over inputs.adonis_bank (paper_banks dirs, k_mu + fs_*, w0).
ACHILLES = selection.oracle_signal over inputs.reference (fs_rich oracle npzs, lep/prot_p4/pi_pid).
Both under the config's SignalDef, absolute nb; observables/edges/chi2/ratio (+ optional data overlay)
through the ONE validated plotting.make_figure -> chi2_ratio_panel.

Style-agnostic by design: it saves PDF+PNG at cfg.out_path and inherits whatever matplotlib rcParams
the CALLER set (the paper suite calls analysis.paper.style.use() first for the serif look).  This keeps
the core driver free of any analysis/ (application-layer) dependency.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from adonis.workflow import selection as SG
from adonis.workflow.plotting import make_figure
from adonis.workflow.data_overlay import load_overlay


def _merge(parts):
    parts = [p for p in parts if len(p["w"])]
    if not parts:
        raise ValueError("selection returned no events on any input")
    keys = list(parts[0])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def build_adonis(cfg):
    banks = cfg.inputs.get("adonis_bank") or []
    if not banks:
        raise ValueError("no ADoNIS input banks (expected inputs.adonis_bank: [<paper_banks dir>, ...])")
    return _merge([SG.bank_signal(p, cfg.signal) for p in banks])


def build_reference(cfg):
    refs = cfg.inputs.get("reference") or []
    if not refs:
        raise ValueError("no ACHILLES reference (expected inputs.reference: [<fs_rich oracle npz>, ...])")
    return _merge([SG.oracle_signal(p, cfg.signal) for p in refs])


def run_analysis(cfg, ado_label="ADoNIS", ref_label="ACHILLES", panel_w=3.4, **style_kw):
    """style_kw (panel_kw / legend_fn / fig_h / title_kw / label_as_xlabel) is forwarded verbatim to
    make_figure; omitted -> the historical look.  Keeps this core driver free of any analysis/ import."""
    ado = build_adonis(cfg); ref = build_reference(cfg)
    specs = [(o.key, o.bin_edges(), o.label) for o in cfg.observables]
    data = load_overlay(cfg, specs)
    # a config may pin where its legend goes (curves differ panel to panel); the caller's legend_fn
    # decides what that means, so this stays a hint rather than matplotlib state in the core driver.
    lf = style_kw.pop("legend_fn", None)
    if lf is not None and getattr(cfg, "legend_loc", ""):
        _inner, _loc = lf, cfg.legend_loc
        lf = lambda ax, has_parts: _inner(ax, has_parts, loc=_loc)
    fig, results, sig = make_figure(specs, ref, ado, title=cfg.title, ratio_band=cfg.ratio_band,
                                    ratio_ylim=cfg.ratio_ylim, ado_label=ado_label,
                                    ref_label=ref_label, data=data, panel_w=panel_w,
                                    legend_fn=lf, **style_kw)
    out = Path(cfg.out_path); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"selected sigma [nb]: ADoNIS {sig['ado']:.4e}  ACH {sig['ref']:.4e}  ACH/ADO {sig['ach_ado']:.3f}")
    print(f"{'var':12s} chi2/ndf  ACH/ADO")
    for key, _, _ in specs:
        r = results[key]
        print(f"{key:12s} {r['chi2']/max(r['ndf'],1):7.2f}   {r['ach_ado']:.3f}")
    print("wrote", out, "(+ .pdf)")
    return {"results": results, "sigma": sig, "out": str(out)}


def main(argv=None):
    import argparse
    from adonis.workflow.config import load_analysis_config
    ap = argparse.ArgumentParser(description="ADoNIS-vs-ACHILLES config-driven figure (one AnalysisConfig YAML).")
    ap.add_argument("config", help="an AnalysisConfig YAML (configs/analysis/*.yaml)")
    ap.add_argument("--out", default=None, help="override out_path")
    a = ap.parse_args(argv)
    cfg = load_analysis_config(a.config)
    if a.out:
        cfg.out_path = a.out
    run_analysis(cfg)


if __name__ == "__main__":
    main()
