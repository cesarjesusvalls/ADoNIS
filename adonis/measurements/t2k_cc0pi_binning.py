"""Published binnings of the T2K CC0pi double-differential measurement (arXiv:2002.09323).

Reference data about a measurement, in the same sense as the data releases next door: the edges are
whatever T2K published, and nothing here may adjust them to suit a fit. Kept separate from
adonis/unfold/binning.py so the general unfolding framework does not carry one experiment's bin edges.

The top edge of every cos(theta_mu) slice is an overflow bin running to 30 GeV/c.
"""
import numpy as np

from adonis.unfold.binning import StaircaseGrid, _subdivide

PI = float(np.pi)

T2K_CC0PI_SLICES = [
    ((-1.00, 0.20), [0.0, 30.0]),
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
