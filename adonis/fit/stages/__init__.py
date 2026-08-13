"""Fit stages: the closure fit and the uncertainty estimators.

Moved here from analysis/paper/physfit so the fit machinery lives in ONE package -- analysis/paper is
for figures, which read persisted npz and never fit.  Module names are unchanged so the move is a pure
relocation; the entry point is `python -m adonis.fit <config> --stage <name>`, never these directly.
"""
