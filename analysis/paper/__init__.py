"""Figures for the ADoNIS methodology paper (docs/paper_plan.md).

One sub-package per paper section. Each is a THIN DRIVER over the reusable adonis/ engines
(adonis.reweight, adonis.fsi.cascade, adonis.flux, ...) -- it must not re-implement physics,
binning, or reweighting:

  sec1_validation  ADoNIS reproduces ACHILLES   (adonis.reweight.bank_plot/bank_reweight + paper.beams)
  sec2_gradients   gradients for all 27 knobs   (adonis.reweight.full_knobs/bank_reweight)
  sec3_fisher      Fisher info per observable
  sec4_closure     closures on fitted knobs
  sec5_unknowns    unknown-unknown taxonomy

Figures land in output/paper/ (gitignored); every one is regenerable from committed code.
"""
