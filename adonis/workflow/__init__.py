"""ADoNIS workflow API: YAML-config-driven bank generation and analysis.

    from adonis.workflow import (load_gen_config, load_analysis_config,
                                 generate_bank, run_analysis)

Generation is ONE backbone: `generate_bank(cfg, outdir)` builds every bank (weak / EM / hadron) from a
single GenConfig, driven by config fields not code paths (adonis.workflow.cli is the thin CLI shell).
Analysis describes input banks, a signal topology, observables+binning, and an optional data overlay.
"""
from adonis.workflow.materials import (NuclearTarget, REGISTRY, UnsupportedMaterial,  # noqa: F401
                                       parse_formula, resolve_targets, stoichiometric_weights)
from adonis.workflow.config import load_gen_config, load_analysis_config  # noqa: F401


def generate_bank(*a, **k):
    from adonis.workflow.generate_bank import generate_bank as _f
    return _f(*a, **k)


def run_analysis(*a, **k):
    from adonis.workflow.analyze import run_analysis as _f
    return _f(*a, **k)
