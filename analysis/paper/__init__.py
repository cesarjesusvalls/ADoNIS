"""Figures for the ADoNIS methodology paper (docs/paper_plan.md).

One sub-package per paper section. Each is a THIN DRIVER over the reusable adonis/ engines
(adonis.reweight, adonis.fsi.cascade, adonis.flux, ...) -- it must not re-implement physics,
binning, or reweighting:

  sec1_validation  ADoNIS reproduces ACHILLES   (adonis.reweight.bank_plot/bank_reweight + paper.beams)
  sec2_fisher      Fisher info per observable   (which knobs the data constrains)
  sec3_gradients   per-bin gradients of the fittable knobs
  sec4_closure     fitting & statistical interpretability (closure, coverage, errors, corner)

Figures land in output/paper/ (gitignored); every one is regenerable from committed code.
"""
