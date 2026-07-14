"""Figures for the ADoNIS methodology paper (docs/paper_plan.md).

One sub-package per paper section. Each is a THIN DRIVER over the validated engines -- it must not
re-implement physics, binning, or reweighting:

  sec1_validation  ADoNIS reproduces ACHILLES   <- analysis.t2k.make_plots, bank_matrix
  sec2_gradients   gradients for all 27 knobs   <- full_knobs.knob_specs, bank_reweight.weight_jit
  sec3_fisher      Fisher info per observable   <- scripts/altgen/physical_fit.py (Gate I)
  sec4_closure     closures on fitted knobs     <- scripts/altgen/physical_fit_run.py
  sec5_unknowns    unknown-unknown taxonomy     <- scripts/altgen/physical_fit_run.py (modes)

Figures land in output/paper/ (gitignored); every one is regenerable from committed code.
"""
