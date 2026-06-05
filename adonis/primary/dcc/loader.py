"""Parser for the ACHILLES DCC electroweak amplitude table `data/dcc_EW.dat`.

This reads the ANL-Osaka DCC single-pion-production amplitudes that ACHILLES ships
as an ASCII table, into structured NumPy arrays. It mirrors the Fortran reader
`src/Achilles/fortran/amp_dcc_sl.f` (`read_amp`).

File layout (ASCII):
  1.  `njLs`                       number of partial waves (14)
  2.  njLs lines: `jpin,Lpin,ispin,itpin`  per wave (2J, 2L, 2Iπ, 2I_total),
      labelled s11 s31 p11 p13 p31 p33 d13 d15 d33 d35 f15 f17 f35 f37
  3.  `maxw mxq2`                  grid sizes (67, 28)
  4.  maxw lines: W grid [MeV]     (first token; trailing 0. ignored)
  5.  mxq2 lines: Q^2 grid [MeV^2]
  6.  `namp1 namp2 namp3`          entry counts for the 3 blocks
  7.  namp1 lines  (VECTOR)        `ie iq idx ipw igmb ils  za1 za2 za3`
  8.  namp2 lines  (ISOSCALAR-V)
  9.  namp3 lines  (AXIAL)         (neutrino case)

Each amplitude line is Fortran format `6i4,10g20.10`: six integer indices then
three COMPLEX `za` = six reals (Re,Im pairs):
  za(1) bare,  za(2) dressed (with N* resonances),  za(3) non-resonant background.
Indices: ie=W index, iq=Q^2 index, idx=current component, ipw=partial wave,
igmb=meson-baryon channel, ils (always 1 here).

We keep all three za components per entry so the cross-section layer can choose the
combination (ACHILLES forms sum_n za(n)*idata(n)). Output arrays are complex,
shaped (n_q2, n_w, n_idx, n_pw, n_gmb, 3).
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_PATH = "/Users/cjesus/Software/DiffSinglePiProd/Achilles/data/dcc_EW.dat"
PW_LABELS = ["s11", "s31", "p11", "p13", "p31", "p33", "d13", "d15",
             "d33", "d35", "f15", "f17", "f35", "f37"]


@dataclass
class DCCTable:
    pw_2J: np.ndarray          # (njLs,) = 2J per partial wave
    pw_2L: np.ndarray          # (njLs,) = 2L
    pw_ispin: np.ndarray       # (njLs,)
    pw_2I: np.ndarray          # (njLs,) = 2*I_total
    W: np.ndarray              # (n_w,) invariant mass grid [MeV]
    Q2: np.ndarray             # (n_q2,) momentum transfer grid [MeV^2]
    vec: np.ndarray            # (n_q2, n_w, n_idx, n_pw, n_gmb, 3) complex
    isv: np.ndarray            # isoscalar-vector, same shape
    axial: np.ndarray          # axial, same shape

    @property
    def labels(self):
        return PW_LABELS[: len(self.pw_2J)]


def _read_block(lines, start, count, dims):
    """Parse `count` amplitude lines into a complex array of shape dims+(3,).

    Returns (array, next_start). Each line: 6 ints + 6 reals (3 complex za)."""
    text = "".join(lines[start:start + count])
    raw = np.loadtxt(io.StringIO(text))
    if raw.ndim == 1:
        raw = raw[None, :]
    idx = raw[:, :6].astype(np.int64) - 1                  # to 0-based
    ie, iq, ic, ipw, igmb, _ils = idx.T
    za = (raw[:, 6:12:2] + 1j * raw[:, 7:12:2])            # (count, 3) complex
    arr = np.zeros(dims + (3,), dtype=np.complex128)
    # filter to in-range indices (Fortran skips igmb>mbs, ipw>njLs)
    ok = (iq < dims[0]) & (ie < dims[1]) & (ic < dims[2]) & \
         (ipw < dims[3]) & (igmb < dims[4]) & (iq >= 0) & (ie >= 0) & (ic >= 0)
    arr[iq[ok], ie[ok], ic[ok], ipw[ok], igmb[ok]] = za[ok]
    return arr, start + count


def parse_dcc_ew(path=DEFAULT_PATH) -> DCCTable:
    with open(path) as f:
        lines = f.readlines()

    p = 0
    njLs = int(lines[p].split()[0]); p += 1
    pw = np.array([[int(x) for x in lines[p + i].split()[0].split(",")]
                   for i in range(njLs)])
    p += njLs
    maxw, mxq2 = (int(x) for x in lines[p].split()[:2]); p += 1
    W = np.array([float(lines[p + i].split()[0]) for i in range(maxw)]); p += maxw
    Q2 = np.array([float(lines[p + i].split()[0]) for i in range(mxq2)]); p += mxq2
    namp = [int(x) for x in lines[p].split()[:3]]; p += 1

    # The component index `idx` is the Fortran ixi1 = photon-polarization x nucleon-
    # helicity (zampv/zmtx first dimension, size 8; see amp_dcc_sl.f read_amp):
    #   ismi(ixi1)=[1,1,0,0,-1,-1,2,2] -> photon pol igm1 in {+1,0,-1,2(charge/time)}
    #   isbi(ixi1)=[1,-1,1,-1,1,-1,1,-1] -> nucleon helicity sign
    # Stored sparsely: vec/isv use idx={1,2,3}, AXIAL adds idx=7 (the charge/PCAC
    # induced-pseudoscalar piece). Sizing n_idx to the vec block's max (3) would
    # silently DROP the axial idx=7 -> always size to the full ixi1 range (8).
    n_idx = 8
    head = np.loadtxt(io.StringIO("".join(lines[p:p + namp[0]])), usecols=(4,))
    n_gmb = int(head.max())          # meson-baryon channel (only piN=1 populated)
    dims = (mxq2, maxw, n_idx, njLs, n_gmb)

    vec, p = _read_block(lines, p, namp[0], dims)
    isv, p = _read_block(lines, p, namp[1], dims)
    axial, p = _read_block(lines, p, namp[2], dims)

    return DCCTable(pw[:, 0], pw[:, 1], pw[:, 2], pw[:, 3], W, Q2, vec, isv, axial)


def load_cached(path=DEFAULT_PATH, cache=None) -> DCCTable:
    """Parse once and cache to .npz next to this module for fast reload."""
    cache = cache or os.path.join(os.path.dirname(__file__), "_dcc_ew_cache.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        d = np.load(cache)
        return DCCTable(d["pw_2J"], d["pw_2L"], d["pw_ispin"], d["pw_2I"],
                        d["W"], d["Q2"], d["vec"], d["isv"], d["axial"])
    t = parse_dcc_ew(path)
    np.savez_compressed(cache, pw_2J=t.pw_2J, pw_2L=t.pw_2L, pw_ispin=t.pw_ispin,
                        pw_2I=t.pw_2I, W=t.W, Q2=t.Q2, vec=t.vec, isv=t.isv, axial=t.axial)
    return t


if __name__ == "__main__":
    t = parse_dcc_ew()
    print(f"partial waves ({len(t.labels)}): {t.labels}")
    print(f"  2J = {t.pw_2J.tolist()}")
    print(f"  2L = {t.pw_2L.tolist()}")
    print(f"  2I = {t.pw_2I.tolist()}")
    print(f"W grid:  {t.W.size} pts, {t.W.min():.1f}..{t.W.max():.1f} MeV")
    print(f"Q^2 grid: {t.Q2.size} pts, {t.Q2.min():.0f}..{t.Q2.max():.0f} MeV^2 "
          f"({t.Q2.max()/1e6:.2f} GeV^2)")
    print(f"vec   shape {t.vec.shape}, nonzero {np.count_nonzero(t.vec)}")
    print(f"isv   shape {t.isv.shape}, nonzero {np.count_nonzero(t.isv)}")
    print(f"axial shape {t.axial.shape}, nonzero {np.count_nonzero(t.axial)}")
    # spot check: first dressed vector entry at (iq=0,ie=0,idx=1,ipw=0,igmb=0)
    print("sample vec[0,0,1,0,0] (bare,dressed,nonres) =", t.vec[0, 0, 1, 0, 0])
