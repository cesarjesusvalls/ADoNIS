"""run_analysis(AnalysisConfig): the analysis driver.

ADoNIS prediction = select_signal applied to each ADoNIS rich engine bank (roles adonis_res /
adonis_qe) under the config's SignalDef, concatenated -- this is the config-driven channel
composition (CC1pi = res + qe-created; CC0pi = qe + res-absorbed).  Reference = select_reference over
the ACHILLES rich bank(s), absolute nb.  Bins/chi2/ratio + optional data overlay via plotting.make_figure.
"""
from __future__ import annotations
import numpy as np
import adonis.workflow.selection as SG
from adonis.workflow.plotting import make_figure
from adonis.workflow.data_overlay import load_overlay

_ADONIS_ROLES = ("adonis_res", "adonis_qe")


def _merge(parts):
    parts = [p for p in parts if len(p["w"])]
    keys = parts[0]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def build_adonis(cfg):
    sd = cfg.signal; parts = []
    for role in _ADONIS_ROLES:
        for path in cfg.inputs.get(role, []):
            parts.append(SG.select_signal(path, sd, carbon_only=cfg.carbon_only))
    if cfg.inputs.get("adonis_h"):
        raise NotImplementedError(
            "adonis_h (free-proton precomputed-observable bank) combination is not wired yet; "
            "current carbon figures are 12C-only.")
    if not parts:
        raise ValueError("no ADoNIS input banks (expected inputs.adonis_res / adonis_qe)")
    return _merge(parts)


def build_reference(cfg):
    refs = [SG.select_reference(p, cfg.signal, carbon_only=cfg.carbon_only)
            for p in cfg.inputs["reference"]]
    return refs[0] if len(refs) == 1 else _merge(refs)


def run_analysis(cfg, ado_label="ENGINE", ref_label="ACHILLES", panel_w=3.4):
    ado = build_adonis(cfg); ref = build_reference(cfg)
    specs = [(o.key, o.bin_edges(), o.label) for o in cfg.observables]
    data = load_overlay(cfg, specs)
    fig, results, sig = make_figure(specs, ref, ado, title=cfg.title, ratio_band=cfg.ratio_band,
                                    ratio_ylim=cfg.ratio_ylim, ado_label=ado_label,
                                    ref_label=ref_label, data=data, panel_w=panel_w)
    fig.savefig(cfg.out_path, dpi=120)
    print(f"selected sigma: ADoNIS {sig['ado']:.4e}  ACH {sig['ref']:.4e}  ACH/ADO {sig['ach_ado']:.3f}")
    print(f"{'var':10s} chi2/ndf  ACH/ADO")
    for key, _, _ in specs:
        r = results[key]
        print(f"{key:10s} {r['chi2']/max(r['ndf'],1):7.2f}   {r['ach_ado']:.3f}")
    print("wrote", cfg.out_path)
    return {"results": results, "sigma": sig, "out": cfg.out_path}
