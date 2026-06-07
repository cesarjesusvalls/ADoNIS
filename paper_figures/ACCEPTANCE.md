# Acceptance criterion (ALL figures)

A paper figure is **reproduced** only when, in **every bin of every panel**:

    | ADoNIS / ACHILLES  -  1 |  <  0.03   (3%)

- The metric is the **per-bin ratio of the two predictions** (ADoNIS vs ACHILLES), with a
  consistent normalisation. It is NOT chi2/ndf, NOT agreement of means, NOT agreement with the
  experimental data. ACHILLES is the target; the data is secondary context.
- ADoNIS and ACHILLES must be produced with the **identical** signal definition / tight
  phase-space cuts / observable, read from the NUISANCE source and the paper (arXiv-2508.19213v2).
- Every `make_figN.py` prints the per-bin ADoNIS/ACHILLES ratio and `max|ratio-1|`; a figure is
  only marked ✅ when that max is < 3% across all bins/panels.

When a bin fails: first rule out ADoNIS statistics (the tight cuts keep ~0.5% of events — need
millions generated), then attack the physics residual (CCQE muon reconstruction, proton-FSI
low-p, spectral function).
