"""Exact ACHILLES constants (Constants.hh), MeV units throughout.  Reproduced to machine
precision so the transliterated cross section is on ACHILLES's absolute nb scale."""
import numpy as np

# Masses [MeV]
mp = 938.27208816
mn = 939.56542054
mN = (mp + mn) / 2.0
mN2 = mN * mN
mpip = 139.57018
mpi0 = 134.9764
mdelta = 1232.25

# hbar c: HBARC = 197.3269804 MeV.fm (Constants.hh comment, = HBAR*C in their unit system);
# HBARC2 = HBARC^2 * 10  [mb MeV^2]  (the *10 converts fm^2 -> mb)
HBARC = 197.3269804
HBARC2 = HBARC * HBARC * 10.0           # mb MeV^2  (= 0.3893793721 mb GeV^2)
TO_NB = 1e6                             # mb -> nb (FluxFactor)

# EW parameters (Constants.hh:39-58)
alpha = 1.0 / 137.035999084
GF = 1.1663787e-5 / 1000.0 / 1000.0     # MeV^-2  ( _GeV = 1000 )
sin2w = 0.23129
cos2w = 1.0 - sin2w
MZ = np.sqrt((np.pi * alpha) / (np.sqrt(2.0) * GF * cos2w * sin2w))
MW = MZ * np.sqrt(cos2w)
GAMZ = 2.4952e3
GAMW = 2.0895e3
Vud = 0.97367
Vus = 0.225
ee = np.sqrt(4.0 * np.pi * alpha)
cw = np.sqrt(cos2w)
sw = np.sqrt(sin2w)
