"""Figures for the ADoNIS methodology paper: one sub-package per topic, each a thin consumer of the
`adonis` package (physics, binning, reweighting, fitting, unfolding live there).

Every sub-package exposes the same entry point:

    python -m analysis.paper.<topic>.make [names...] [--label L]

Sub-packages: validation (ADoNIS vs ACHILLES), grad_info (Fisher info + gradient shapes), inference
(fits, closure, coverage, corner plots), unfolding, performance.  `style` holds the shared look and is
the only module the sub-packages import from each other.

Figures land in output/paper/ (gitignored).
"""
