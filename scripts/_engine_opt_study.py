"""(n_workers, n_w, P) cascade throughput study.  Spawns _engine_opt_worker subprocesses (each JITs
independently, like the real multi-process overnight setup) and reports, per configuration, the
TOTAL JIT time and the ev/s.

  Part A (single process): sweep P x n_w  -> per-process JIT(s), run(s), ev/s, overflow%.
  Part B (workers):        at a chosen (P, n_w), sweep n_workers in parallel -> total JIT (max across the
                           workers, since they compile concurrently), aggregate ev/s = (W*N_per)/wall.

ev/s is pure-run throughput; "total JIT" is wall time lost to compilation (one-time per process).

Run: python -u scripts/_engine_opt_study.py [N=12000] [Pgrid=12,16,24] [nwgrid=0,512,1024,2048,4096] [Klist=1,2,4,8] [bestP=16] [bestNW=1024]
"""
import sys, os, time, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(os.path.dirname(HERE), ".venv/bin/python")
N = sys.argv[1] if len(sys.argv) > 1 else "12000"
PG = (sys.argv[2] if len(sys.argv) > 2 else "12,16,24").split(",")
NWG = (sys.argv[3] if len(sys.argv) > 3 else "0,512,1024,2048,4096").split(",")
KL = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4,8").split(",")]
BESTP = sys.argv[5] if len(sys.argv) > 5 else "16"
BESTNW = sys.argv[6] if len(sys.argv) > 6 else "1024"


def worker(P, nw, seed=0):
    """Run one subprocess; return its RESULT dict (or None)."""
    p = subprocess.run([PY, "-u", os.path.join(HERE, "_engine_opt_worker.py"), N, str(P), str(nw), "none", str(seed)],
                       capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    sys.stderr.write(p.stdout[-500:] + p.stderr[-500:]); return None


def main():
    print(f"=== Part A: single-process P x n_w  (N={N}) ===", flush=True)
    print(f"{'P':>4} {'n_w':>6} {'JIT(s)':>8} {'run(s)':>8} {'ev/s':>9} {'ofl%':>7}", flush=True)
    for P in PG:
        for nw in NWG:
            r = worker(P, nw)
            if r:
                print(f"{P:>4} {nw:>6} {r['jit_s']:>8} {r['run_s']:>8} {r['evps']:>9} {r['ofl_pct']:>7}", flush=True)

    print(f"\n=== Part B: workers at P={BESTP} n_w={BESTNW}  (each worker N={N}) ===", flush=True)
    print(f"{'K':>3} {'wall(s)':>9} {'totJIT(s)':>10} {'agg ev/s':>10} {'perW ev/s':>10}", flush=True)
    for K in KL:
        t0 = time.time()
        procs = [subprocess.Popen([PY, "-u", os.path.join(HERE, "_engine_opt_worker.py"), N, BESTP, BESTNW, "none", str(s)],
                                  stdout=subprocess.PIPE, text=True) for s in range(K)]
        outs = [p.communicate()[0] for p in procs]
        wall = time.time() - t0
        rs = []
        for o in outs:
            for ln in o.splitlines():
                if ln.startswith("RESULT "):
                    rs.append(json.loads(ln[7:]))
        if not rs:
            print(f"{K:>3}  (no results)", flush=True); continue
        nper = rs[0]["N"]; agg = K * nper / wall; perw = sum(r["evps"] for r in rs) / len(rs)
        totjit = max(r["jit_s"] for r in rs)          # workers compile concurrently -> wall JIT ~ the max
        print(f"{K:>3} {wall:>9.1f} {totjit:>10.1f} {agg:>10.0f} {perw:>10.0f}", flush=True)


if __name__ == "__main__":
    main()
