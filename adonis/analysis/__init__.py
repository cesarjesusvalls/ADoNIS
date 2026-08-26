"""Sample-centric analysis core.

A `sample` (bank + signal + observables) is the fundamental object; the paper's sections are verbs
over it -- plot (sec1), gradient (sec2/3), fit (sec4/5). `AnaSample` (sample.py) is that object, built
from an `adonis.workflow.config.AnalysisConfig`; `knobs.py` is the shared reweight-knob basis.
Configs live in configs/samples/; analysis/paper/ holds only thin callers.
"""
