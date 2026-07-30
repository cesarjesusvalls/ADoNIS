"""ADoNIS free-nucleon RES single-pion sigma(E_nu) SCAN BANK -- the ADoNIS-side mirror of the
ACHILLES `output/oracle_freenucleon_scan/`.

WHY THIS EXISTS.  Fig 2 is the only paper figure whose ADoNIS side was *generated at plot time*:
3 channels x 10 energies x n samples of the DCC amplitude, ~2 h per run, every time anyone touched
the plot style.  The ACHILLES side of the same figure has always been a frozen scan on disk, read in
~0.1 s.  This restores the symmetry: generate once, read forever.

ONE NPZ PER POINT, exactly like the ACHILLES scan:

    <out>/<channel>_E<MeV>.npz     e.g. adonis_freenucleon_scan/ppip_E1000.npz

so the 30 points are independent: a run can be interrupted and resumed, sharded across jobs, or have
a single point regenerated without touching the others.  Each file stores the reduced result
(sigma + its MC standard error, in nb) plus the exact parameters it was produced with, so a stale
point is detectable rather than silently reused.

  python -m analysis.paper.freenucleon_bank                  # fill in whatever is missing
  python -m analysis.paper.freenucleon_bank --jobs 6         # 30 independent points, in parallel
  python -m analysis.paper.freenucleon_bank --n 200000       # match the ACHILLES 200k stats
  python -m analysis.paper.freenucleon_bank --force          # regenerate everything

Cost is entirely `adonis.channels.res.sigma_free_nucleon` -> exclusive_amps2_batch (~400 ev/s), so
wall time is ~n*30/400 s divided by --jobs.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The scan definition lives HERE (bank side), and fig2 imports it, so the generator and the figure
# can never disagree about which points the figure is made of.
ENERGIES = np.array([400, 600, 800, 1000, 1300, 1600, 2000, 2600, 3400, 4400.0])   # MeV
# tag -> (res.CHANNELS index, tex label, ACHILLES scan species, ACHILLES pi_pid)
# res.CHANNELS order is [n->n pi+, n->p pi0, p->p pi+]; the paper plots p pi+, n pi+, n pi0.
CHANNEL_SPECS = [
    ("ppip", 2, r"$\nu_\mu\, p\to\mu^- p\,\pi^+$", "H", 211),
    ("npip", 0, r"$\nu_\mu\, n\to\mu^- n\,\pi^+$", "N", 211),
    ("npi0", 1, r"$\nu_\mu\, n\to\mu^- p\,\pi^0$", "N", 111),
]
DEFAULT_OUT = ROOT / "output" / "adonis_freenucleon_scan"
DEFAULT_N = 80_000


def point_path(out, tag, E):
    return Path(out) / f"{tag}_E{int(E)}.npz"


def _seed(ci, j):
    """Per-point MC seed.  NB this keys on the res.CHANNELS index, whereas the old plot-time scan keyed
    on the channel's position in the figure's own list -- so two of the three channels draw a different
    (equally valid) sample than before, and their sigma moves by O(SE).  Any fixed convention is fine;
    it just must not be mistaken for a physics change."""
    return 1000 * ci + j


def _compute(tag, ci, j, E, n):
    """One scan point.  Imported lazily and inside the worker so --jobs can spawn cleanly."""
    import adonis.channels.dcc.current as _dcc
    _dcc.BATCH_INTERP = "spline"                 # jitted spline == ACHILLES interpolate_amp
    from adonis.channels.res import sigma_free_nucleon, CHANNELS
    s, e = sigma_free_nucleon(float(E), CHANNELS[ci], n=int(n), seed=_seed(ci, j))
    return s, e


def _worker(args):
    tag, ci, j, E, n, out = args
    s, e = _compute(tag, ci, j, E, n)
    p = point_path(out, tag, E)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")       # atomic: a killed run leaves no half-written point
    np.savez(tmp, sigma_nb=np.array(s), err_nb=np.array(e), E_MeV=np.array(float(E)),
             n=np.array(int(n)), seed=np.array(_seed(ci, j)), channel=np.array(tag),
             chan_index=np.array(int(ci)))
    os.replace(tmp, p)
    return tag, float(E), s, e


def is_current(out, tag, E, n):
    """True if the stored point was made with this n (and is readable).  Statistics are the only
    thing that changes the answer here -- the physics inputs are compiled in."""
    p = point_path(out, tag, E)
    if not p.exists():
        return False
    try:
        d = np.load(p, allow_pickle=False)
        return int(d["n"]) == int(n)
    except Exception:
        return False


def load_scan(out=DEFAULT_OUT, require=True):
    """{tag: (sigma_nb[nE], err_nb[nE])} over ENERGIES.

    Two ways this refuses to hand back a figure that would quietly lie:
      * a MISSING point raises with the exact build command (never silently regenerate for hours);
      * MIXED statistics raise too -- an interrupted `--n` bump leaves some points at the old n and
        some at the new one, and a plot mixing them looks perfectly normal while being inconsistent.
    """
    missing, res, ns = [], {}, {}
    for tag, _ci, _tex, _sp, _pid in CHANNEL_SPECS:
        s = np.full(len(ENERGIES), np.nan); e = np.full(len(ENERGIES), np.nan)
        for j, E in enumerate(ENERGIES):
            p = point_path(out, tag, E)
            if not p.exists():
                missing.append(p.name); continue
            d = np.load(p, allow_pickle=False)
            s[j] = float(d["sigma_nb"]); e[j] = float(d["err_nb"]); ns[p.name] = int(d["n"])
        res[tag] = (s, e)
    if missing and require:
        raise FileNotFoundError(
            f"{len(missing)} free-nucleon scan point(s) missing from {out} (e.g. {missing[0]}).\n"
            f"Generate with:  python -m analysis.paper.freenucleon_bank --jobs 6")
    uniq = sorted(set(ns.values()))
    if len(uniq) > 1:
        by = {u: [k for k, v in ns.items() if v == u] for u in uniq}
        raise ValueError(
            f"free-nucleon scan in {out} has MIXED statistics {uniq} -- an interrupted --n change.\n"
            + "\n".join(f"  n={u}: {len(by[u])} point(s), e.g. {by[u][0]}" for u in uniq)
            + f"\nFinish it with:  python -m analysis.paper.freenucleon_bank --n {uniq[-1]} --jobs 6")
    return res


def scan_n(out=DEFAULT_OUT):
    """The n every point in the bank was built with (None if empty).  load_scan guarantees it is one value."""
    for tag, _ci, _tex, _sp, _pid in CHANNEL_SPECS:
        for E in ENERGIES:
            p = point_path(out, tag, E)
            if p.exists():
                return int(np.load(p, allow_pickle=False)["n"])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="MC samples per point")
    ap.add_argument("--jobs", type=int, default=1, help="parallel points (they are independent)")
    ap.add_argument("--force", action="store_true", help="regenerate points that already exist")
    ap.add_argument("--only", default=None,
                    help="rebuild just these points: comma-separated <tag> or <tag>_E<MeV> "
                         "(e.g. ppip, or ppip_E400) -- the points are independent")
    a = ap.parse_args(argv)

    sel = None if not a.only else set(x.strip() for x in a.only.split(","))
    todo = [(tag, ci, j, float(E), a.n, a.out)
            for (tag, ci, _tex, _sp, _pid) in CHANNEL_SPECS
            for j, E in enumerate(ENERGIES)
            if (sel is None or tag in sel or f"{tag}_E{int(E)}" in sel)
            and (a.force or not is_current(a.out, tag, E, a.n))]
    have = len(CHANNEL_SPECS) * len(ENERGIES) - len(todo)
    print(f"free-nucleon scan -> {a.out}   n={a.n}  present={have}  to build={len(todo)}", flush=True)
    if not todo:
        return
    if a.jobs > 1:
        # spawn (not fork): the workers import jax themselves, and forking an initialised jax is unsafe
        import multiprocessing as mp
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")        # one BLAS thread each, else the pool oversubscribes
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for tag, E, s, e in pool.imap_unordered(_worker, todo):
                print(f"  {tag:5s} E={E:6.0f}  sigma={s:.4e} nb  SE={100*e/max(s,1e-300):.2f}%", flush=True)
    else:
        for t in todo:
            tag, E, s, e = _worker(t)
            print(f"  {tag:5s} E={E:6.0f}  sigma={s:.4e} nb  SE={100*e/max(s,1e-300):.2f}%", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
