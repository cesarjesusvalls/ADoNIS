"""Figures for the ADoNIS methodology paper.

One sub-package per topic.  Each is a THIN CONSUMER of the reusable `adonis` package: the physics,
binning, reweighting, fitting and unfolding all live there, and what lives here is the choice of what
to plot and how it should look.  A module here that re-implements physics is a bug -- it means the
capability was written in the wrong layer.

Every sub-package exposes the SAME entry point, so there is exactly one way to build any figure:

    python -m analysis.paper.<topic>.make [names...] [--label L]

  validation    ADoNIS reproduces ACHILLES, sample by sample.  Each figure is a YAML spec in the
                directory; the specs are the definition, `helper.render` is the only drawing code.
  grad_info     what the gradients tell you before any fit: Fisher information per observable subset
                (which knobs the data can constrain) and the per-bin gradient shapes (where that
                information comes from).  Both are views of ONE Jacobian.
  inference     fitting and its statistical interpretation: closure, per-sample rates, coverage,
                and the corner comparison of four constructions of the same uncertainty.
  unfolding     the unfolding demonstration -- pure consumers of one npz from `analysis.campaign.unfold_run`.
  performance   what the derivatives and the hardware actually cost: minimiser scaling with and
                without gradients, and event-generation throughput on CPU against GPU.

`style` is the shared look (palette, panel geometry, knob LaTeX names, save paths) and is the only
module the sub-packages import from each other.

Figures land in output/paper/ (gitignored); every one is regenerable from committed code plus the
banks and fit artefacts named in its config.
"""
