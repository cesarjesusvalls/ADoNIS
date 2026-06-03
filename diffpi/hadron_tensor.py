"""Full DCC hadron-tensor assembly -- decoded conventions (Phase-2.5, milestone 6a).

This module is the scaffold for the faithful port of amp_dcc_sl.f::amplitude(): it
turns the knob-scaled DCC partial-wave amplitudes into the helicity current
zj_mu(isf, isi, mu) and, contracted with the lepton tensor, the differential cross
section -- replacing the angle-integrated diagonal-bilinear approximation (dcc_xsec).

Milestone 6a DECODE (from read_amp + amplitude() in amp_dcc_sl.f):

* The amplitude file dcc_EW.dat stores, per (W-index ie, Q^2-index iq), entries
  `idx ipw igmb ils  za(1) za(2) za(3)` with the physical amplitude = sum_n za(n)
  (bare + dressed-N* + non-resonant; idata=(1,1,1)).

* `idx` is the Fortran ixi1 = (photon polarization) x (nucleon helicity), the first
  dimension (size 8) of zampv/zmtx. The decode arrays:
      isbi[ixi1] = nucleon-helicity sign      = [+1,-1,+1,-1,+1,-1,+1,-1]
      ismi[ixi1] = photon polarization igm1   = [ 1, 1, 0, 0,-1,-1, 2, 2]
      ismix[ixi1]                              = [ 1, 1, 0, 0,-1,-1, 0, 0]
  so igm1 in {+1 (transv), 0 (long), -1 (transv), 2 (charge/time)}.
  Stored sparsely: VEC and ISV use idx in {1,2,3}; AXIAL adds idx=7 (the charge/time
  component = induced-pseudoscalar / PCAC piece absent from the conserved EM vector
  current). The remaining ixi1 (4,5,6,8) are reconstructed by parity/helicity
  symmetry inside amplitude() (factors fn_sign, the ixi_cnv reordering).

* `ipw` = partial wave 1..14 (rows with ipw>njLs=14 are dropped); `igmb` = meson-
  baryon channel (only piN=1 populated); `ils` = 1 (single L-S coupling here).

* Partial-wave quantum numbers (jpind,Lpind,ispind,itpind) = (2J, 2L, 2Ipi, 2Itot):
      pw  1 s11 (2J1 2L0 2I1)      8 d15 (2J5 2L4 2I1)
      pw  2 s31 (2J1 2L0 2I3)      9 d33 (2J3 2L4 2I3)
      pw  3 p11 (2J1 2L2 2I1)     10 d35 (2J5 2L4 2I3)
      pw  4 p13 (2J3 2L2 2I1)     11 f15 (2J5 2L6 2I1)
      pw  5 p31 (2J1 2L2 2I3)     12 f17 (2J7 2L6 2I1)
      pw  6 p33 (2J3 2L2 2I3) Delta 13 f35 (2J5 2L6 2I3)
      pw  7 d13 (2J3 2L4 2I1)     14 f37 (2J7 2L6 2I3)

* Helicity -> Cartesian current (amplitude() zjx_mu block), igm1 spherical -> mu:
      zj_mu[0] = zcrnt(igm1=0  longitudinal-time slot stored as 0)
      zj_mu[3] = zcrnt(igm1=2  charge/z)
      zj_mu[1] = (zcrnt(-1) - zcrnt(+1)) / sqrt(2)
      zj_mu[2] = (zcrnt(-1) + zcrnt(+1)) * i / sqrt(2)
  followed by a Lorentz boost xlr (lorentz_trans) from the piN-CM to the lab frame.

* Cross section (gamma*N -> piN, from the dsigma/dOmega comment block):
      dsigma/dOmega = sum_{si,sf,lambda} (4pi^2 alpha / E_gamma) |zj_mu|^2
                      * m_N k_pi / (16 pi^3 W) / 4
  For neutrino CC the photon flux is replaced by the V-A lepton tensor (with the
  vector-axial INTERFERENCE term, absent from the diagonal approximation).

STILL TO PORT (6b/6c): Wigner-d (setdfun/dfun), the two frame rotations (rspin ->
zmxpi/zmxpf), isospin Clebsch-Gordan (cbg), associated Legendre (ylmsub/bleg) and
the azimuthal phases, the helicity->spin conversion, the Lorentz boost, and the
lepton-tensor contraction (sigma_T + sigma_L; V-A for CC).

VALIDATION (6d): integrated over the pion solid angle, this MUST collapse to the
diagonal-bilinear assembly in dcc_xsec (the regression check), and then close the
residual peak gap vs the EM/CC oracles (dcc_fold_validation.png).
"""
from __future__ import annotations

import numpy as np

# --- decoded index conventions (1-based ixi1 = file idx) --------------------- #
# index 0 unused (pad) so ISBI[ixi1] reads with the Fortran 1-based ixi1.
ISBI = np.array([0, 1, -1, 1, -1, 1, -1, 1, -1])      # nucleon-helicity sign
ISMI = np.array([0, 1, 1, 0, 0, -1, -1, 2, 2])        # photon polarization igm1
ISMIX = np.array([0, 1, 1, 0, 0, -1, -1, 0, 0])
IXI_CNV = np.array([0, 1, 2, 5, 6, 3, 4, 7, 8])       # ixi1p -> ixi1 reordering

# photon-polarization values actually stored per current:
VEC_IDX = (1, 2, 3)            # igm1 = +1(hel+), +1(hel-), 0(hel+)
AXIAL_IDX = (1, 2, 3, 7)       # adds igm1 = 2 (charge/time, PCAC)

PW_LABELS = ("s11", "s31", "p11", "p13", "p31", "p33",
             "d13", "d15", "d33", "d35", "f15", "f17", "f35", "f37")
