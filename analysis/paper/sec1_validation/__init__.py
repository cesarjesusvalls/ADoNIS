"""Section-1 / appendix validation suite: ADoNIS reproduces ACHILLES on the paper's figures.

Config-driven: one AnalysisConfig YAML per figure in configs/analysis/, run through the ONE driver
(adonis.workflow.analyze.run_analysis) under the paper style.  Add a figure by dropping a YAML --
no code change.  Replaces the old frozen-bank sec1_validation (kept as old_sec1_validation).
"""
