"""Section-1 / appendix validation suite: ADoNIS reproduces ACHILLES on the paper's figures.

Config-driven: one AnalysisConfig YAML per figure in configs/analysis/, run through the ONE driver
(adonis.workflow.analyze.run_analysis) under the paper style, plus the standalone drivers (fig1/2/3/456/10/13)
for the non-histogram observables.  Add a config figure by dropping a YAML -- no code change.
"""
