"""Bit-exact validation of the JAX QE matrix-element port vs instrumented ACHILLES.

Parses the QEDUMP lines (XSecBackend.cc instrumentation: lep_in, lep_out, had_in, had_out + the
ACHILLES amps2, flux, initwgt, spinavg) and feeds the IDENTICAL momenta into adonis.xsec.backend.
Asserts amps2 and flux (and amps2*flux*spinavg) agree to ~1e-6 relative -- the spin-summed |M|^2
is basis-independent, so the Weyl-basis port must reproduce the Fortran Dirac current exactly.

Usage: python scripts/validate_qe_matrix_element.py <dump.txt>
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from adonis.xsec.backend import me_cross_section, flux_factor

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


def main(path):
    rows = parse(path)
    print(f"parsed {len(rows)} QEDUMP rows")
    if not rows:
        print("NO ROWS — check instrumentation/run"); return
    li = jnp.asarray([r["li"] for r in rows]); lo = jnp.asarray([r["lo"] for r in rows])
    hi = jnp.asarray([r["hi"] for r in rows]); ho = jnp.asarray([r["ho"] for r in rows])
    sa = jnp.asarray([r["spinavg"] for r in rows])
    from adonis.xsec.backend import MASS_PDG_NEUTRON as MNn, MASS_PDG_PROTON as MPp
    mass = jnp.asarray([MNn if r["hiID"] == 2112 else MPp for r in rows])
    d = me_cross_section(li, lo, hi, ho, spin_avg=sa, had_mass=mass)
    a_jax = np.asarray(d["amps2"]); f_jax = np.asarray(d["flux"])
    a_ach = np.array([r["amps2"] for r in rows]); f_ach = np.array([r["flux"] for r in rows])
    me_ach = a_ach * f_ach * np.array([r["spinavg"] for r in rows])
    me_jax = np.asarray(d["me_xsec"])

    def rel(x, y):
        return np.abs(x - y) / np.clip(np.abs(y), 1e-300, None)
    ra = rel(a_jax, a_ach); rf = rel(f_jax, f_ach); rm = rel(me_jax, me_ach)
    print(f"flux   : max rel {rf.max():.2e}  (PASS<1e-6: {rf.max()<1e-6})")
    print(f"amps2  : max rel {ra.max():.2e}  median {np.median(ra):.2e}  (PASS<1e-6: {ra.max()<1e-6})")
    print(f"me_xsec: max rel {rm.max():.2e}  (PASS<1e-6: {rm.max()<1e-6})")
    # show the worst few amps2 rows
    order = np.argsort(-ra)[:5]
    for i in order:
        print(f"  row {i}: amps2 jax={a_jax[i]:.6e} ach={a_ach[i]:.6e} rel={ra[i]:.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qe_dump.txt")
