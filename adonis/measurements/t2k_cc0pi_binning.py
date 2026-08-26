"""Published binnings of the T2K CC0pi double-differential measurement (arXiv:2002.09323).

Reference data about a measurement, in the same sense as the data releases next door: the edges are
whatever T2K published, and nothing here may adjust them to suit a fit.  They lived in
adonis/unfold/binning.py, which made a general unfolding framework carry one experiment's bin edges.

The top edge of every cos(theta_mu) slice is an OVERFLOW bin running to 30 GeV/c.
"""
import numpy as np

from adonis.unfold.binning import StaircaseGrid, _subdivide

PI = float(np.pi)

# T2K CC0pi DOUBLE-DIFFERENTIAL BINNING (arXiv:2002.09323), for the lepton-kinematics unfolding.
#
# WHY A STAIRCASE AND NOT A RECTANGLE.  p_mu and cos theta_mu are strongly correlated -- fast muons are
# forward -- so a rectangular grid built from 1-D marginal quantiles empties its corners: measured on
# this bank, a 5 x 6 rectangle put 0.0 events in the high-p_mu / backward cell.  A template with no
# events is identically zero, its c_j is unconstrained, and the fit is singular.  The published analysis
# solves this by binning p_mu CONDITIONALLY on the cos theta_mu slice, and these are its edges.
#
# p_mu is in GeV/c (the unit of the measurement); cos theta_mu is dimensionless.  58 truth bins.
#
# RESOLUTION, against the section's detector (sigma_p = 10%, sigma_theta = 5 deg): 54 of the 58 bins have
# width > resolution.  The binding constraint is now the ANGLE, not the momentum -- the cos slices
# spanning 0.80-0.94 come out at ratio 1.01-1.18 -- which is the opposite of the STV grid, where every
# delta-p_T bin was below 1.0.  One p_mu bin (3.0-3.25 GeV inside cos 0.94-0.98) is genuinely
# unresolved at ratio 0.80; it sits in the far tail and holds almost nothing.
# EVERY SLICE'S TOP BIN IS AN OVERFLOW TO 30 GeV/c, not a closed edge.  Checked against the published
# table for cos theta_mu in [0.94, 0.98], whose last bin is 3 -> 30 rather than 3 -> 3.25: its value,
# 4.698e-41, sits ~100x below its neighbours, which is what dividing by a width of 27 instead of ~1
# produces.  Closing those axes at the second-to-last edge pushed every high-momentum signal event
# outside the truth grid, where build() correctly but silently reclassifies it as BACKGROUND -- an
# event with no template to scale it.  The measurement loses no rate; the fit did.
T2K_CC0PI_SLICES = [
    ((-1.00, 0.20), [0.0, 30.0]),                                               # backward catch-all
    (( 0.20, 0.60), [0.0, 0.3, 0.4, 0.5, 0.6, 30.0]),
    (( 0.60, 0.70), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 30.0]),
    (( 0.70, 0.80), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 30.0]),
    (( 0.80, 0.85), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 30.0]),
    (( 0.85, 0.90), [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 30.0]),
    (( 0.90, 0.94), [0.0, 0.4, 0.5, 0.6, 0.8, 1.25, 2.0, 30.0]),
    (( 0.94, 0.98), [0.0, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 30.0]),
    (( 0.98, 1.00), [0.0, 0.5, 0.7, 0.9, 1.25, 2.0, 3.0, 5.0, 30.0]),
]



def t2k_truth_grid():
    return StaircaseGrid(T2K_CC0PI_SLICES, "t2k-truth")


def t2k_reco_grid(k=2):
    return StaircaseGrid(_subdivide(T2K_CC0PI_SLICES, k=k), f"t2k-reco-x{k}")
