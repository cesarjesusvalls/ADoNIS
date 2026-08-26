"""Aggregate the (N, n) workload surface from the per-event-count bench_fair runs.

Medians are over the NOISE REALISATIONS, not over timing repeats.  The dominant spread in this study is
seed-to-seed, not run-to-run: Gauss-Newton's iteration count on the same problem varied 14 -> 44 across
realisations at n=17, because the Hessian term it drops is proportional to the residual and the residual
is what the throw changes.  Quoting a timing spread here would advertise a precision the measurement
does not have, so the min-max across seeds is carried alongside every median.

Asimov is reported separately and never pooled with the noise cells: on a perfect closure the dropped
term vanishes and Gauss-Newton *becomes* Newton, so mixing the two would flatter it.

    python -m analysis.benchmarks.bench_report [--glob output/altgen/bench_fair_amp_N*.npz]
"""
from __future__ import annotations
import argparse, glob as globmod, sys
import numpy as np

METH = ("gn", "migrad+g", "migrad")


def load(paths):
    rows = []
    for p in sorted(paths):
        z = np.load(p, allow_pickle=True)
        rows += list(z["rows"])
    return rows


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def med(rows, key):
    v = [r[key] for r in rows]
    return (np.median(v), min(v), max(v)) if v else (np.nan,) * 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="output/altgen/bench_fair_amp_N*.npz")
    a = ap.parse_args(argv)
    paths = globmod.glob(a.glob)
    if not paths:
        raise SystemExit(f"no files matching {a.glob}")
    rows = load(paths)
    Ns = sorted({r["N"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    print(f"loaded {len(rows)} fits from {len(paths)} files")
    print(f"  events/sample : {Ns}   (resident = 10x)")
    print(f"  dials         : {ns}")
    seeds = sorted({r["tag"] for r in rows if r["tag"] != "asimov"})
    print(f"  realisations  : asimov + {len(seeds)} noise\n")

    for case, tags in (("NOISE (primary)", seeds), ("ASIMOV (contrast)", ["asimov"])):
        print(f"================ {case} ================")
        for what, key, fmt in (("wall-clock seconds", "wall", "8.2f"),
                               ("event-passes", "passes", "8.0f"),
                               ("objective evaluations", "nfev", "8.0f")):
            print(f"\n---- {what} (median over realisations) ----")
            print(f"{'n':>3} " + " ".join(f"{m:>26}" for m in METH))
            print(f"{'':>3} " + " ".join(f"{'  '.join(f'{N//1000}k' for N in Ns):>26}" for m in METH))
            for n in ns:
                cells = []
                for m in METH:
                    per = []
                    for N in Ns:
                        rr = [r for r in sel(rows, n=n, N=N, method=m) if r["tag"] in tags]
                        per.append(f"{med(rr, key)[0]:{fmt}}" if rr else "       -")
                    cells.append(" ".join(per))
                print(f"{n:>3} " + " | ".join(cells))

        # ratios at the largest available N
        Nb = Ns[-1]
        print(f"\n---- ratio to Gauss-Newton at {Nb:,} events/sample ----")
        print(f"{'n':>3} {'gn wall':>9} {'gn passes':>10} " +
              " ".join(f"{m+' x wall':>16} {m+' x pass':>16}" for m in METH[1:]))
        for n in ns:
            g = [r for r in sel(rows, n=n, N=Nb, method="gn") if r["tag"] in tags]
            if not g:
                continue
            gw, gp = med(g, "wall")[0], med(g, "passes")[0]
            cells = []
            for m in METH[1:]:
                rr = [r for r in sel(rows, n=n, N=Nb, method=m) if r["tag"] in tags]
                cells.append(f"{med(rr,'wall')[0]/gw:16.2f} {med(rr,'passes')[0]/gp:16.2f}" if rr
                             else f"{'-':>16} {'-':>16}")
            print(f"{n:>3} {gw:9.2f} {gp:10.0f} " + " ".join(cells))

        # seed-to-seed spread: the honest error bar on everything above
        print(f"\n---- seed spread at n={ns[-1]}, {Nb:,} events/sample (min-max over realisations) ----")
        for m in METH:
            rr = [r for r in sel(rows, n=ns[-1], N=Nb, method=m) if r["tag"] in tags]
            if not rr:
                continue
            w, wlo, whi = med(rr, "wall")
            p, plo, phi = med(rr, "passes")
            f, flo, fhi = med(rr, "nfev")
            print(f"  {m:>9}: wall {w:6.2f}s [{wlo:.2f}-{whi:.2f}]   "
                  f"passes {p:6.0f} [{plo:.0f}-{phi:.0f}]   nfev {f:5.0f} [{flo:.0f}-{fhi:.0f}]")
        print()

    # scaling in N, per method, at the largest n -- fitted on the medians
    print("================ SCALING ================")
    nb = ns[-1]
    print(f"wall-clock vs events, n={nb}, noise cells: fit log(t) = alpha log(N) + c")
    for m in METH:
        xs, ys = [], []
        for N in Ns:
            rr = [r for r in sel(rows, n=nb, N=N, method=m) if r["tag"] in seeds]
            if rr:
                xs.append(np.log(N)); ys.append(np.log(med(rr, "wall")[0]))
        if len(xs) >= 2:
            al, c = np.polyfit(xs, ys, 1)
            print(f"  {m:>9}: alpha = {al:5.2f}   (1.0 = linear in events)")
    print(f"\nevent-passes vs dials, at {Ns[-1]:,} events/sample, noise cells")
    for m in METH:
        xs, ys = [], []
        for n in ns:
            rr = [r for r in sel(rows, n=n, N=Ns[-1], method=m) if r["tag"] in seeds]
            if rr:
                xs.append(np.log(n)); ys.append(np.log(med(rr, "passes")[0]))
        if len(xs) >= 2:
            be, c = np.polyfit(xs, ys, 1)
            print(f"  {m:>9}: beta  = {be:5.2f}   (1.0 = linear in dials)")


if __name__ == "__main__":
    sys.exit(main())
