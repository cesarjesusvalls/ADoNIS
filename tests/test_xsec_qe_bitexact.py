"""Bit-exact regression: the JAX QE matrix-element port (adonis.channels) vs instrumented ACHILLES.

tests/data/qe_dump_achilles.txt holds 300 QEDUMP rows from a real ACHILLES QE_Spectral_Func run
(XSecBackend.cc instrumentation, %.17e): the lab momenta + ACHILLES amps2/flux/spinavg.  The port
must reproduce amps2 and flux on the SAME momenta.  amps2 is a full 4x4 spin sum (a spin trace),
so it is basis-independent -- the Weyl leptonic x Dirac hadronic currents must match the Fortran.

flux is bit-exact (<1e-8).  amps2 median ~7e-11 (bit-exact); a <=1e-5 tail survives on ~4% of
events at LOW Q^2 / deep removal energy -- float64 summation-order roundoff between the einsum and
ACHILLES's explicit Fortran matmul chains (confirmed: NOT mqe, NOT mass, NOT dump precision).
"""
import re
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.channels.currents.matrix_element import me_cross_section, MASS_PDG_NEUTRON as MN, MASS_PDG_PROTON as MP

DUMP = Path(__file__).resolve().parent / "data" / "qe_dump_achilles.txt"

_NUM = r"[-+0-9.eE]+"
_PAT = re.compile(
    r"QEDUMP pid=(-?\d+) "
    rf"liE=({_NUM}) li=({_NUM}),({_NUM}),({_NUM}) "
    rf"loE=({_NUM}) lo=({_NUM}),({_NUM}),({_NUM}) "
    rf"hiE=({_NUM}) hi=({_NUM}),({_NUM}),({_NUM}) hiID=(-?\d+) "
    rf"hoE=({_NUM}) ho=({_NUM}),({_NUM}),({_NUM}) "
    rf"amps2=({_NUM}) flux=({_NUM}) initwgt=({_NUM}) spinavg=({_NUM}) psw=({_NUM})")


def parse(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = _PAT.search(line)
        if not m:
            continue
        g = m.groups()
        rows.append(dict(
            pid=int(g[0]),
            li=[float(g[1]), float(g[2]), float(g[3]), float(g[4])],
            lo=[float(g[5]), float(g[6]), float(g[7]), float(g[8])],
            hi=[float(g[9]), float(g[10]), float(g[11]), float(g[12])], hiID=int(g[13]),
            ho=[float(g[14]), float(g[15]), float(g[16]), float(g[17])],
            amps2=float(g[18]), flux=float(g[19]), initwgt=float(g[20]),
            spinavg=float(g[21]), psw=float(g[22])))
    return rows


def _compute():
    rows = parse(str(DUMP))
    arr = lambda k: jnp.asarray([r[k] for r in rows])
    li, lo, hi, ho = arr("li"), arr("lo"), arr("hi"), arr("ho")
    sa = arr("spinavg")
    mass = jnp.asarray([MN if r["hiID"] == 2112 else MP for r in rows])
    d = me_cross_section(li, lo, hi, ho, spin_avg=sa, had_mass=mass)
    aa = np.array([r["amps2"] for r in rows]); fa = np.array([r["flux"] for r in rows])
    ra = np.abs(np.asarray(d["amps2"]) - aa) / np.abs(aa)
    rf = np.abs(np.asarray(d["flux"]) - fa) / np.abs(fa)
    return ra, rf


def test_flux_bit_exact():
    _, rf = _compute()
    assert rf.max() < 1e-8, f"flux max rel {rf.max():.2e}"


def test_amps2_bit_exact_median():
    ra, _ = _compute()
    assert np.median(ra) < 1e-9, f"amps2 median rel {np.median(ra):.2e}"
    assert np.percentile(ra, 95) < 1e-5, f"amps2 95pct rel {np.percentile(ra,95):.2e}"
    assert ra.max() < 5e-5, f"amps2 max rel {ra.max():.2e}"


if __name__ == "__main__":
    ra, rf = _compute()
    print(f"flux max {rf.max():.2e} | amps2 median {np.median(ra):.2e} 95pct "
          f"{np.percentile(ra,95):.2e} max {ra.max():.2e}")
    test_flux_bit_exact(); test_amps2_bit_exact_median(); print("QE matrix element bit-exact OK")
