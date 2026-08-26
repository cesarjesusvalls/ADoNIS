"""Single source of truth for physical constants — verbatim ACHILLES values.

Two families, BOTH verbatim from the ACHILLES source. Never "fix" one into the other:
the generator itself uses both, and bit-exact reproduction requires matching each code
path's own choice.

* `Constants.hh` precision values (lowercase: ``mp``, ``mn``, ``HBARC``, ``GF``, ...) —
  used by the transliterated cross-section code (``adonis/channels``) and the Oset port.
* `Particles.yml` rounded masses (``MASS_PDG_*``) — used wherever ACHILLES goes through
  ``ParticleInfo::Mass()`` (FluxFactor's struck-nucleon mass, channel rest masses, ...).

The uppercase aliases (``M_P``, ``MQE``, ``M_PI``, ``W_THR``, ...) serve the
differentiable DCC path (hard cuts from ``currents_pi_dcc.f90``).
``adonis/channels/constants.py`` re-exports the lowercase family for back-compat.
Mass-convention ROLES (which mass plays which part in RES) live in
``adonis/primary/dcc/conventions.py``, not here.
"""
import numpy as np

# --- include/Achilles/Constants.hh (precise) ------------------------------------------
mp = 938.27208816         # proton mass  [MeV]
mn = 939.56542054         # neutron mass [MeV]
me = 0.51099895000        # electron mass [MeV] (PDG); used by the (e,e') reconstruction (E_QE, p_e)
mN = (mp + mn) / 2.0      # = 938.91875435 ; Constant::mN / mqe (main_xsec_new.f90:90)
mN2 = mN * mN
mpip = 139.57018
mpi0 = 134.9764
mdelta = 1232.25

# hbar c: HBARC = 197.3269804 MeV.fm (Constants.hh); HBARC2 = HBARC^2 * 10 [mb MeV^2]
HBARC = 197.3269804
HBARC2 = HBARC * HBARC * 10.0           # mb MeV^2  (= 0.3893793721 mb GeV^2)
TO_NB = 1e6                             # mb -> nb (FluxFactor)

# EW parameters (Constants.hh:39-58)
alpha = 1.0 / 137.035999084
GF_GEV = 1.1663787e-5                   # GeV^-2
GF = GF_GEV / 1000.0 / 1000.0           # MeV^-2  ( _GeV = 1000 )
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

# --- data/Particles.yml (rounded; what ParticleInfo::Mass() returns) ------------------
MASS_PDG_PROTON = 938.27
MASS_PDG_NEUTRON = 939.57
MASS_PDG_PIP = 139.571
MASS_PDG_PI0 = 134.977
MASS_PDG_MUON = 105.7

# --- DCC-path aliases + hard cuts (currents_pi_dcc.f90) -------------------------------
M_P = mp
M_N = mn
M_NUC = mN                # mqe = 0.5*(mp+mn)
MQE = M_NUC               # struck-nucleon energy: E_struck = mqe - E_removal
M_PI0 = mpi0
M_PIP = mpip
M_PI = (2 * M_PIP + M_PI0) / 3.0   # = 138.039 ; DCC fpio=(2 f_pi+ + f_pi0)/3 (pion_init)

W_THR = 1076.957          # piN threshold cut [MeV] (= M_NUC + M_PI to 3 dp)
W_MAX = 2000.0            # upper W cut [MeV]
Q2_MAX = 5.0e6            # upper Q^2 cut [MeV^2]  (Q2 in [0, 5 GeV^2])

# --- nuclear-target masses (T2K TKI reconstruction: 12C -> p + 11B) -------------------
M_12C = 11174.862         # 12C nuclear mass [MeV]
M_11B = 10252.547         # 11B residual mass [MeV]

# --- forward-angle cut sentinel (T2K signal defs) ------------------------------------
# EXACT expression matching workflow.config / workflow.plotting / reweight.bank_plot (bit-identical);
# oracle.extract used np.cos(70*pi/180) which can differ by 1 ulp -- it now shares this value.
COS70 = float(np.cos(np.deg2rad(70.0)))

# --- PDG id sets (membership tests) ---------------------------------------------------
PDG_NEUTRINOS = frozenset({12, 14, 16, -12, -14, -16})
PDG_CHARGED_LEPTONS = frozenset({11, 13, -11, -13})
PDG_PIONS = frozenset({111, 211, -211})
PDG_NUCLEONS = frozenset({2112, 2212})
PDG_MESONS = frozenset({111, 211, -211, 221, 130, 310, 311, 321, -321, -311})


# ---- run-mode flags -------------------------------------------------------------------------------- #
# ADONIS_EB_MIRROR: with the mirror on, the spectral-function response is EVEN in Eb_shift, so a
# negative value denotes the same physical shift |Eb| and the fit must not be bounded at zero.  Two
# modules must agree on it -- adonis.analysis.knobs (is Eb_shift bounded) and adonis.reweight.sf_reweight
# (is the response mirrored) -- so the read happens once, here.  It cannot live in knobs.py because
# sf_reweight cannot import knobs.
import os as _os
EB_MIRROR = _os.environ.get("ADONIS_EB_MIRROR", "") == "1"
