"""ADoNIS workflow API: YAML-config-driven generation and analysis.

    from adonis.workflow import (load_gen_config, load_analysis_config,
                                 run_generation, run_analysis)

Generation describes a target material, channels (QE/RES), cascade hyperparameters and tracking;
analysis describes input banks, a signal topology, observables+binning, and an optional data overlay
-- both as standardized YAML files run by the thin drivers scripts/adonis_{generate,analyze}.py.
"""
from adonis.workflow.materials import (NuclearTarget, REGISTRY, UnsupportedMaterial,  # noqa: F401
                                       parse_formula, resolve_targets, stoichiometric_weights)
from adonis.workflow.config import load_gen_config, load_analysis_config  # noqa: F401


def run_generation(*a, **k):
    from adonis.workflow.generate import run_generation as _f
    return _f(*a, **k)


def run_analysis(*a, **k):
    from adonis.workflow.analyze import run_analysis as _f
    return _f(*a, **k)
