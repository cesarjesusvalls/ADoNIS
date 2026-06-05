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

DONE (6b): the knob-independent angular primitives below -- wigner_d_half/int,
cbg, legendre_ylm -- validated vs closed forms (validate_hadron_tensor.py).

interpolate_amp DECODE (the zampv -> zmtx step, lines 588-960; for 6c):
 (1) EW isospin combination: from the stored vector (zampv) and isoscalar-vector
     (zampv_is), form  isovector=(V-IS)/2, isoscalar=(V+IS)/2  (per idx,pw,gmb).
 (2) Current conservation (vector only): the charge/time component (idx 7,8) is
     DERIVED from the longitudinal (idx 3,4) via V_z = V_0 * omega_cm/q_cm
     (xxx=qc0/qc, qc0=(W^2-m^2-Q^2)/2W). Hence vector stores only idx={1,2,3};
     axial is NOT conserved so it stores idx=7 explicitly (PCAC).
 (3) Parity/helicity symmetry fill: id1=(1,2,3,7) <-> id2=(6,5,4,8) -- the negative-
     photon-helicity / opposite-nucleon-helicity components (idx 4,5,6,8) are filled
     from the stored (idx 1,2,3,7) with a parity phase (zmtx(idxx)=zzz*pha).
 (4) (W,Q^2) bilinear interpolation onto the event point.

STILL TO PORT (6c): interpolate_amp (1)-(4) above; the assembly loop (zfac_a isospin
CG x zmtx, zzz = spin-orbit CG x Legendre x azimuth, accumulate zcrnt); zjx_mu
helicity->Cartesian; the lepton-tensor contraction (sigma_T + sigma_L; V-A for CC).
SIMPLIFICATIONS for the unpolarized, angle-integrated xsec (dsigma/dW,dQ^2): the spin
rotations rspin + the irot_q nucleon-helicity rotation are UNITARY and CANCEL in the
spin sum; dsigma/dW,dQ^2 are Lorentz-invariant so the lab boost (lorentz_trans) is
unneeded -> work in piN-CM with photon along z (irot_q=0 branch). Then
  W^{mu,nu}(W,Q2) = sum_{pw,pw'} A_pw^* G^{mu,nu}_{pw,pw'} A_pw'
with G a knob-independent kernel = integral over Omega_pi of the (validated) angular
functions; off-diagonal pw interference vanishes on angle-integration.

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


# --- Wigner-d functions (port of setdfun + fblmmx) --------------------------- #
from math import factorial as _fact, sqrt as _sqrt


def wigner_d_int(l, mf, mi, cc, ss):
    """Integer Wigner small-d d^l_{mf,mi} via the explicit sum (Fortran fblmmx).
    cc=cos(theta/2), ss=sin(theta/2).  h(k)=sqrt(k!) in the Fortran common block."""
    if l < 0 or abs(mf) > l or abs(mi) > l:
        return 0.0
    jmip, jmim, jmfp, jmfm = l + mi, l - mi, l + mf, l - mf
    mfmim = mf - mi
    iicos, iisin = 2 * l - mfmim, mfmim
    kmin, kmax = max(0, -mfmim), min(jmip, jmfm)
    num = _sqrt(_fact(jmip) * _fact(jmim) * _fact(jmfp) * _fact(jmfm))
    s = 0.0
    for k in range(kmin, kmax + 1):
        den = _fact(jmip - k) * _fact(k) * _fact(jmfm - k) * _fact(k + mfmim)
        s += (-1) ** (mfmim + k) * num / den * cc ** (iicos - 2 * k) * ss ** (iisin + 2 * k)
    return s


def wigner_d_half(two_j, two_mf, two_mi, x):
    """Half-integer Wigner small-d d^{J}_{mf,mi}(theta) with J=two_j/2,
    mf=two_mf/2, mi=two_mi/2, x=cos(theta).  Port of setdfun's per-(lx,mf,mi) build
    (lx=two_j, mf=two_mf, mi=two_mi in the Fortran 2*projection convention)."""
    lx, mf, mi = two_j, two_mf, two_mi
    ss = _sqrt((1.0 - x) / 2.0)
    cc = _sqrt((1.0 + x) / 2.0)
    jj = (lx - 1) // 2
    mfm, mfp = (mf - 1) // 2, (mf + 1) // 2
    mim, mip = (mi - 1) // 2, (mi + 1) // 2
    out = 0.0
    if jj >= abs(mfm) and jj >= abs(mim):
        out += wigner_d_int(jj, mfm, mim, cc, ss) * _sqrt((lx + mf) * (lx + mi)) / (2 * lx) * cc
    if jj >= abs(mfp) and jj >= abs(mip):
        out += wigner_d_int(jj, mfp, mip, cc, ss) * _sqrt((lx - mf) * (lx - mi)) / (2 * lx) * cc
    if jj >= abs(mfm) and jj >= abs(mip):
        out += -wigner_d_int(jj, mfm, mip, cc, ss) * _sqrt((lx + mf) * (lx - mi)) / (2 * lx) * ss
    if jj >= abs(mfp) and jj >= abs(mim):
        out += wigner_d_int(jj, mfp, mim, cc, ss) * _sqrt((lx - mf) * (lx + mi)) / (2 * lx) * ss
    return out


def _bb(n, k):
    """n!/k! -- ylmsub's LOCAL bb table (bc(k1)/bc(k2))."""
    return _fact(n) / _fact(k)


def _comb(n, k):
    if k < 0 or k > n:
        return 0
    return _fact(n) // (_fact(k) * _fact(n - k))


def _bbg(p, q):
    """The /fdbn/ bb table used by cbg (bifc): C(p,q) if p>=q, else sqrt(C(q,p))."""
    if p >= q:
        return float(_comb(p, q))
    return _sqrt(_comb(q, p))


def cbg(a, x, b, y, c, z):
    """Clebsch-Gordan <a x; b y | c z> (port of Fortran cbg; a,b,c,x,y,z may be
    half-integer). Returns 0 if the selection rules fail."""
    de = 0.01
    if max(abs(a - b) - c, c - a - b, abs(x + y - z), x - a, -x - a, y - b, -y - b) > de:
        return 0.0
    ja, jb, jc = int(a + de - x), int(b + de + y), int(c + de + z)
    ka, kb, kc = int(a + de + a), int(b + de + b), int(c + de + c)
    iss = int(a + de + b + c)
    ia, ib, ic = iss - ka, iss - kb, iss - kc
    lmin, lmax = max(0, ja - ib, jb - ia), min(ic, ja, jb)
    s = 0.0
    for l in range(lmin, lmax + 1):
        s += (-1) ** l * _bbg(ic, l) * _bbg(ib, ja - l) * _bbg(ia, jb - l)
    s *= (_sqrt((kc + 1.0) / (iss + 1.0)) * _bbg(ib, kc) * _bbg(ic, ka)
          / (_bbg(kb, iss) * _bbg(ja, ka) * _bbg(jb, kb) * _bbg(jc, kc)))
    return s


def legendre_ylm(lmax, z):
    """Normalized associated Legendre / real spherical-harmonic theta-part
    Y_l^m(theta, 0) for l=0..lmax, m=-l..l (port of Fortran ylmsub).  z=cos(theta).
    Returns bleg[l, m] indexed as bleg[l, m_index] with m_index in 0..2*lmax (m=m_index-lmax)."""
    import numpy as _np
    bleg = _np.zeros((lmax + 1, 2 * lmax + 1))

    def setb(l, m, v):
        bleg[l, m + lmax] = v

    def getb(l, m):
        return bleg[l, m + lmax]

    z1 = _sqrt(1.0 - z ** 2) + 1e-20
    z2 = z / z1
    setb(0, 0, 1.0)
    for l in range(1, lmax + 1):
        setb(l, l, z1 ** l / (2 ** l) * _bb(2 * l, l))
        setb(l, l - 1, z2 * getb(l, l))
        if l == 1:
            continue
        for m in range(l - 2, -1, -1):
            setb(l, m, (-getb(l, m + 2) + 2 * (m + 1) * z2 * getb(l, m + 1))
                  / ((l - m) * (l + m + 1)))
    import math as _m
    for l in range(lmax + 1):
        for m in range(l + 1):
            fac = _sqrt((2 * l + 1) / (4.0 * _m.pi) * _bb(l - m, l + m)) * (-1) ** m
            setb(l, m, getb(l, m) * fac)
            setb(l, -m, getb(l, m) * (-1) ** m)
    return bleg
