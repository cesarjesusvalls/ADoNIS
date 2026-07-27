"""Back-compat shim: this module was renamed to adonis.reweight.reweight_model (it builds the full
differentiable model; the knob SCHEMA lives in adonis.core.params).  Kept so the frozen golden-gate
script + any external scripts that `import adonis.reweight.full_knobs` keep working."""
from adonis.reweight.reweight_model import (  # noqa: F401
    PhysicsParams, nominal_knobs, knob_specs, build_hv_sf, model_hist_full, _NPW, _EB_EPS, _DELTA_WAVE)
