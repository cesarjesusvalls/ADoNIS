"""B: compare ACHILLES's instrumented OsetCrossSection::AbsCrossSection output (OSETABS stderr
lines from the cascade-instr image) against the ADoNIS port at the IDENTICAL inputs.  If they
agree, the per-nucleon sigma_abs is faithful and the ~25% transparency over-absorption is purely
transport (-> justifies the discrete-geometry rebuild); if not, it's a port bug to fix."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import adonis.fsi.oset_xsec as ox

pat = re.compile(r"OSETABS Tpi=([\d.eE+-]+) pmom=([\d.eE+-]+) dens=([\d.eE+-]+) fermi=([\d.eE+-]+) "
                 r"vrel=([\d.eE+-]+) sqrts=([\d.eE+-]+) cms_mom=([\d.eE+-]+) "
                 r"pabs=([\d.eE+-]+) sabs=([\d.eE+-]+) tot=([\d.eE+-]+)")


def main(path):
    rows = []
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            rows.append([float(x) for x in m.groups()])
    rows = np.array(rows)
    if not len(rows):
        print("no OSETABS lines found in", path); return
    Tpi, pmom, dens, fermi, vrel, sqrts, cms, pabs_a, sabs_a, tot_a = rows.T
    # recover pion mass per line from (pmom, Tpi): pE = Tpi + m,  pE^2 = pmom^2 + m^2
    mpi = (pmom ** 2 - Tpi ** 2) / (2.0 * np.clip(Tpi, 1e-6, None))
    mpi = np.clip(mpi, 130.0, 145.0)                       # charged/neutral pion band
    pE = Tpi + mpi
    tot_d = np.asarray(ox.abs_cross_section(pE, mpi, pmom, vrel, fermi, dens))
    good = (tot_a > 1e-6) & np.isfinite(tot_d)
    r = tot_d[good] / tot_a[good]
    print(f"OSETABS samples: {len(rows)} (used {good.sum()})")
    print(f"ADoNIS/ACHILLES sigma_abs ratio: median={np.median(r):.4f}  mean={np.mean(r):.4f}  "
          f"[{np.percentile(r,5):.3f}, {np.percentile(r,95):.3f}]")
    # show a few representative lines
    print(f"\n{'Tpi':>7} {'dens':>8} {'vrel':>7} {'ACH_tot':>9} {'ADO_tot':>9} {'ratio':>6}")
    idx = np.argsort(Tpi[good])[:: max(1, good.sum() // 12)]
    Tg, dg, vg, ta, td = Tpi[good], dens[good], vrel[good], tot_a[good], tot_d[good]
    for i in idx:
        print(f"{Tg[i]:7.1f} {dg[i]:8.5f} {vg[i]:7.3f} {ta[i]:9.3f} {td[i]:9.3f} {td[i]/ta[i]:6.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_oracle_out/osetabs.log")
