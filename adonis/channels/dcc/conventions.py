"""Single source of truth for RES (single-pion) mass conventions + the ACHILLES-match toggle.

Every value here is a physical input (a PDG/ACHILLES mass) or a derived combination of them --
never a fitted calibration. Imported by both the fast event generator (adonis.channels.res_xsec /
dcc_current) and the differentiable JAX path (adonis.channels.dcc channel/structure), so the two
implementations cannot drift on conventions.

The pion mass plays three distinct roles, matching ACHILLES's own conventions:
  * amplitude-internal (build_zmtx qc / pion-pole facpp):  m_pi = 138.04  (isospin-avg fpio).
  * final-state on-shell + 3-body phase space:             m_pi = 134.98  (mpi0, ALL channels;
        ACHILLES puts every outgoing pion, including pi+, on-shell at mpi0).
  * the strictly-physical per-channel mass (M_PIP / M_PI0)  -- used only when MATCH_ACHILLES=False.

Nucleon mass has two roles:
  * amplitude-internal (qc, qc0):                 m_N = 938.919  (mp+mn)/2 average (== ACHILLES MQE).
  * absolute-norm _NORM = 2pi/(|FResV|^2 (2 m_N)^2):  m_N = 939.566 neutron (ACHILLES Fortran xmn).
"""
from adonis.channels import constants as C

# --- physical masses [MeV]: PDG / ACHILLES Constants.hh -- inputs, not fits -----------------------
M_P   = C.mp            # 938.272  proton
M_N   = C.mn            # 939.565  neutron
M_NUC = C.mN            # 938.919  (mp+mn)/2  -- amplitude-internal nucleon mass
M_PIP = C.mpip          # 139.570  charged pion
M_PI0 = C.mpi0          # 134.976  neutral pion
M_PI_AMP = (2.0 * M_PIP + M_PI0) / 3.0   # 138.039  isospin-avg fpio -- amplitude-internal pion pole

# --- the ACHILLES-match toggle -------------------------------------------------------------------
# True  -> reproduce ACHILLES's (model) mass conventions (mpi0 kinematics, neutron norm mass).
# False -> use strictly-physical per-channel masses; the RES sigma then sits ~1% above ACHILLES.
# Deliberate "match the generator" switch, not physics -- flip it to move away from ACHILLES.
MATCH_ACHILLES = True

def amp_m_N():
    """Amplitude-internal nucleon mass (build_zmtx qc/qc0), both engines."""
    return M_NUC

def amp_m_pi():
    """Amplitude-internal pion mass (build_zmtx pion-pole facpp): isospin-avg fpio, 138.04 MeV."""
    return M_PI_AMP

def norm_m_N():
    """Nucleon mass in _NORM = 2pi/(|FResV|^2 (2 m_N)^2): neutron (ACHILLES xmn)."""
    return M_N

def kin_m_pi(physical_mpi):
    """Final-state on-shell + 3-body phase-space pion mass: mpi0 to match ACHILLES, else physical."""
    return M_PI0 if MATCH_ACHILLES else physical_mpi
