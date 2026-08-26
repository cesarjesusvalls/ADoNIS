"""Physical constants, verbatim from ACHILLES.

Two families, both verbatim from the ACHILLES source; never "fix" one into the other,
since the generator uses both and bit-exact reproduction requires matching each code
path's own choice.

* `Constants.hh` precision values (lowercase: ``mp``, ``mn``, ``HBARC``, ``GF``, ...) —
  used by the transliterated cross-section code (``adonis/channels``) and the Oset port.
* `Particles.yml` rounded masses (``MASS_PDG_*``) — used wherever ACHILLES goes through
  ``ParticleInfo::Mass()`` (FluxFactor's struck-nucleon mass, channel rest masses, ...).

The uppercase aliases (``M_P``, ``MQE``, ``M_PI``, ``W_THR``, ...) serve the
differentiable DCC path (hard cuts from ``currents_pi_dcc.f90``).
``adonis/channels/constants.py`` re-exports the lowercase family.
Mass-convention ROLES (which mass plays which part in RES) live in
``adonis/primary/dcc/conventions.py``, not here.
"""
import numpy as np

mp = 938.27208816
mn = 939.56542054
me = 0.51099895000
mN = (mp + mn) / 2.0
mN2 = mN * mN
mpip = 139.57018
mpi0 = 134.9764
mdelta = 1232.25

HBARC = 197.3269804
HBARC2 = HBARC * HBARC * 10.0
TO_NB = 1e6

alpha = 1.0 / 137.035999084
GF_GEV = 1.1663787e-5
GF = GF_GEV / 1000.0 / 1000.0
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

MASS_PDG_PROTON = 938.27
MASS_PDG_NEUTRON = 939.57
MASS_PDG_PIP = 139.571
MASS_PDG_PI0 = 134.977
MASS_PDG_MUON = 105.7

M_P = mp
M_N = mn
M_NUC = mN
MQE = M_NUC
M_PI0 = mpi0
M_PIP = mpip
M_PI = (2 * M_PIP + M_PI0) / 3.0

W_THR = 1076.957
W_MAX = 2000.0
Q2_MAX = 5.0e6

M_12C = 11174.862
M_11B = 10252.547

COS70 = float(np.cos(np.deg2rad(70.0)))

PDG_NEUTRINOS = frozenset({12, 14, 16, -12, -14, -16})
PDG_CHARGED_LEPTONS = frozenset({11, 13, -11, -13})
PDG_PIONS = frozenset({111, 211, -211})
PDG_NUCLEONS = frozenset({2112, 2212})
PDG_MESONS = frozenset({111, 211, -211, 221, 130, 310, 311, 321, -321, -311})


import os as _os
EB_MIRROR = _os.environ.get("ADONIS_EB_MIRROR", "") == "1"
