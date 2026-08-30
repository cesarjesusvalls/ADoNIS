"""Fit and uncertainty machinery: the kernels a fit run is built from.

The runner that drives them is the caller's: analysis.campaign.run.

  config   FitConfig -- the typed run definition (configs/fits/*.yaml)
  stages   the closure fit and the uncertainty estimators
  nuts     the NUTS sampler; a stage like the others

analysis/paper is for FIGURES, which read persisted npz and never fit.  Nothing in this package should
import from there.
"""
