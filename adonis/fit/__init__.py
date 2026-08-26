"""Fit and uncertainty machinery: one config, one entry point.

    python -m adonis.fit configs/fits/<study>.yaml --stage profile2d --shard i/N

  config   FitConfig -- the typed run definition (configs/fits/*.yaml)
  stages   the closure fit and the uncertainty estimators
  nuts     the NUTS sampler; a stage like the others

analysis/paper is for FIGURES, which read persisted npz and never fit.  Nothing in this package should
import from there.
"""
