"""Single source of truth for RES (single-pion) mass conventions + the ACHILLES-match toggle.

Every value here is a physical input (a PDG/ACHILLES mass) or a derived combination of them --
never a fitted calibration. Imported by both the event generator (adonis.channels.res,
adonis.channels.dcc.current) and the differentiable path (adonis.channels.dcc.channel,
adonis.channels.dcc.structure), so the two implementations cannot drift on conventions.

The pion mass plays three distinct roles, matching ACHILLES's own conventions:
  * amplitude-internal (build_zmtx qc / pion-pole facpp): isospin-avg (fpio).
  * final-state on-shell + 3-body phase space: mpi0, ALL channels (ACHILLES puts
        every outgoing pion, including pi+, on-shell at mpi0).
  * the strictly-physical per-channel mass (M_PIP / M_PI0)  -- used only when MATCH_ACHILLES=False.

Nucleon mass has two roles:
  * amplitude-internal (qc, qc0):                 (mp+mn)/2 average (== ACHILLES MQE).
  * absolute-norm _NORM = 2pi/(|FResV|^2 (2 m_N)^2):  neutron mass (ACHILLES Fortran xmn).
"""
from adonis.channels import constants as C

M_P   = C.mp
M_N   = C.mn
M_NUC = C.mN
M_PIP = C.mpip
M_PI0 = C.mpi0
M_PI_AMP = (2.0 * M_PIP + M_PI0) / 3.0

MATCH_ACHILLES = True

def amp_m_N():
    """Amplitude-internal nucleon mass (build_zmtx qc/qc0), both engines."""
    return M_NUC

def amp_m_pi():
    """Amplitude-internal pion mass (build_zmtx pion-pole facpp): isospin-avg fpio."""
    return M_PI_AMP

def norm_m_N():
    """Nucleon mass in _NORM = 2pi/(|FResV|^2 (2 m_N)^2): neutron (ACHILLES xmn)."""
    return M_N

def kin_m_pi(physical_mpi):
    """Final-state on-shell + 3-body phase-space pion mass: mpi0 to match ACHILLES, else physical."""
    return M_PI0 if MATCH_ACHILLES else physical_mpi
