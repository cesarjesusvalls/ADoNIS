"""Fit stages: the closure fit and the uncertainty estimators.

analysis/paper is for figures, which read persisted npz and never fit; the fit machinery lives here.
Entry point is `python -m analysis.campaign.run <config> --stage <name>`, never these modules
directly.
"""
