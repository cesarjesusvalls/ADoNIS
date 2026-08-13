"""Sec-4 diagnostics: the analyses behind the coverage argument.

These are not paper figures.  They are the measurements that decided what figures A-D say, and they are
here because each produced a number that is quoted:

  toys_vs_profile  toy / profile / Laplace / NUTS intervals per dial
  toy_restart      refit every toy from a second start -- 1.2% of blind fits worse by >1 in chi2
  mirror_scan      is the binned prediction two-to-one in S_Delta?  (no: the per-event vertex is
                   Q^2-dependent, so binning destroys the degeneracy)
  slice_sdelta     where the tail toys map in every other dial
  avg_data         the mean per-bin data fluctuation of a toy subgroup -- what selected them
  hist2d_raw       the raw NUTS histograms behind the corner contours
  binscan          contour jitter vs bin count, which set the adaptive binning
  central          central vs shortest intervals: the shortest one is mode-seeking and unstable
  occam            does profile x sqrt(det V_nuis) reproduce the NUTS marginal?  (yes, to 0.03)
  diag_panels      profile vs NUTS 68% areas per corner panel

They still read absolute scratchpad-free paths only via sys.path; several hardcode the sec4_P1 label.
Parameterising them belongs with P2 of docs/consolidation_plan.md, not with vendoring them.
"""
