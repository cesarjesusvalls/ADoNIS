"""Figure 11 -- MicroBooNE NC 1pi0 Xp DOUBLE-differential on 40Ar: d2sigma/dcos(theta_pi0) dp_pi0.

A standalone driver rather than a plain config, because it is the one Section 1 figure whose panels
are SLICES of one observable rather than different observables.  Everything else -- selection,
chi2/ratio panels, absolute normalisation -- goes through the same validated machinery as every other
figure; only the spec list is built by slicing.

    python -m analysis.paper.sec1_validation.fig11_nc_doublediff [--slices 4] [--n-bins 16]

ADoNIS vs ACHILLES, NO data overlay, like every other Section 1 figure.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from adonis.workflow.config import load_analysis_config          # noqa: E402
from adonis.workflow import selection as SG                      # noqa: E402
from adonis.workflow.plotting import make_figure                 # noqa: E402
from analysis.paper import style                                 # noqa: E402

CFG = "analysis/paper/figures/fig11_uboone_nc1pi0_doublediff.yaml"


def _slice(sel, lo, hi):
    """Restrict a selection dict to a cos(theta_pi0) slice, keeping every parallel array aligned."""
    m = (sel["cos_pi0"] >= lo) & (sel["cos_pi0"] < hi)
    return {k: (np.asarray(v)[m] if np.ndim(v) and len(np.asarray(v)) == len(m) else v)
            for k, v in sel.items()}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", type=int, default=4, help="number of cos(theta_pi0) slices")
    ap.add_argument("--n-bins", type=int, default=16, help="p_pi0 bins per slice")
    ap.add_argument("--max-cols", type=int, default=2, help="panels per row (row wrapping)")
    a = ap.parse_args(argv)

    style.use()
    cfg = load_analysis_config(str(ROOT / CFG))
    ado = SG.bank_signal_nc(cfg.inputs["adonis_bank"][0], cfg.signal)
    ref = SG.oracle_signal_nc(cfg.inputs["reference"][0], cfg.signal)

    edges_c = np.linspace(-1.0, 1.0, a.slices + 1)
    p_edges = np.linspace(0.0, 1000.0, a.n_bins + 1)

    # One spec per slice.  Each panel gets its OWN pre-sliced selection, so the shared machinery does
    # not need to know about slicing at all -- it just sees `slices` different observables.
    specs, ado_parts, ref_parts = [], {}, {}
    for i in range(a.slices):
        lo, hi = edges_c[i], edges_c[i + 1]
        key = f"p_pi0_s{i}"
        specs.append((key, p_edges, rf"$p_{{\pi^0}}$ [MeV/c],  ${lo:.2f}<\cos\theta_{{\pi^0}}<{hi:.2f}$"))
        for src, dst in ((ado, ado_parts), (ref, ref_parts)):
            s = _slice(src, lo, hi)
            dst[key] = s["p_pi0"]
            dst.setdefault("w", []).append(s["w"])
            dst.setdefault("chan", []).append(s["chan"])
    # make_figure wants ONE dict with per-key value arrays and a single 'w'; give each panel its own
    # weights by carrying them per key.  The shared panel builder reads sel[key] and sel["w"], so the
    # cleanest correct thing is to call it once per slice-set with aligned arrays.
    n_tot = sum(len(v) for v in ado_parts["w"])
    if n_tot == 0:
        raise SystemExit("no NC1pi0 events selected -- is the bank generated and merged?")

    figs = []
    for i, (key, edges, label) in enumerate(specs):
        lo, hi = edges_c[i], edges_c[i + 1]
        a_s, r_s = _slice(ado, lo, hi), _slice(ref, lo, hi)
        a_s = dict(a_s); r_s = dict(r_s)
        a_s[key] = a_s["p_pi0"]; r_s[key] = r_s["p_pi0"]
        figs.append((key, edges, label, a_s, r_s))

    # single figure, one column per slice, wrapped by --max-cols
    merged_ado = {"w": np.concatenate([f[3]["w"] for f in figs]),
                  "chan": np.concatenate([f[3]["chan"] for f in figs])}
    merged_ref = {"w": np.concatenate([f[4]["w"] for f in figs]),
                  "chan": np.concatenate([f[4]["chan"] for f in figs])}
    # per-key values must align with the concatenated weights: pad each key with NaN outside its slice
    off_a = np.cumsum([0] + [len(f[3]["w"]) for f in figs])
    off_r = np.cumsum([0] + [len(f[4]["w"]) for f in figs])
    for i, (key, edges, label, a_s, r_s) in enumerate(figs):
        va = np.full(len(merged_ado["w"]), np.nan); va[off_a[i]:off_a[i + 1]] = a_s[key]
        vr = np.full(len(merged_ref["w"]), np.nan); vr[off_r[i]:off_r[i + 1]] = r_s[key]
        merged_ado[key] = va; merged_ref[key] = vr

    fig, results, sig = make_figure(
        [(k, e, l) for k, e, l, _, _ in figs], merged_ref, merged_ado,
        title=cfg.title, ratio_band=tuple(cfg.ratio_band), ratio_ylim=tuple(cfg.ratio_ylim),
        ado_label="ADoNIS", ref_label="ACHILLES", panel_w=2.6, fig_h=3.2, min_w=3.2,
        panel_kw=style.panel_kw(ratio_yticks=[0.8, 1.0, 1.2]), legend_fn=style.panel_legend,
        label_as_xlabel=True, title_kw={"fontsize": 9, "y": 0.985, "va": "top"},
        rect_top=0.97, max_cols=a.max_cols)

    out = ROOT / cfg.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")
    for k, r in results.items():
        print(f"  {k}: chi2/ndf = {r['chi2']/max(r['ndf'],1):.2f}  ACH/ADO = {r['ach_ado']:.4f}")
    return 0


def render(spec=None):
    """Figure-hook entry (analysis/paper/figures). Compute is main()'s, unchanged; params from the spec
    map onto main()'s CLI so the standalone and hook paths run identical code."""
    p = (spec or {}).get("params", {}) or {}
    return main(["--slices", str(p.get("slices", 4)),
                 "--n-bins", str(p.get("n_bins", 16)),
                 "--max-cols", str(p.get("max_cols", 2))])


if __name__ == "__main__":
    sys.exit(main())
