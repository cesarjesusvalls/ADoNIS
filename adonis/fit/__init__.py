"""Fit and uncertainty machinery: one config, one entry point.

    python -m adonis.fit configs/fits/<study>.yaml --stage profile2d --shard 3/36

  config   FitConfig -- the typed run definition (configs/fits/*.yaml)
  stages   the closure fit and the uncertainty estimators
  nuts     the exact NUTS sampler (it is a stage, but predates the package layout)

analysis/paper is for FIGURES, which read persisted npz and never fit.  Nothing in this package should
import from there.
"""
