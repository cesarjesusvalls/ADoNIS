"""Decisive amplitude bench: parse ACHILLES RESGEN dump (ach_amps2 at physical kinematics)
for nu p -> mu- n pi+ mono 1 GeV, recompute my_amps2 on the SAME events, bin ratio my/ach
in Q^2 and W.  This is the falsifiable test bed -- toggle a hypothesis in build_zmtx, rerun,
watch the low-Q^2 ratio move toward 1.0.  Chunked + live progress.

Usage: python scripts/resgen_ratio.py [N] [nchunks]   (small N first as a smoke test)."""
import sys, re, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
from adonis.xsec.dcc_current import exclusive_amps2_batch

LOG = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles/_resrun_out/reslog_H_mono_gen.txt")


def _kv(tok):
    """'key=val' -> (key, val_string)."""
    k, v = tok.split("=", 1)
    return k, v


def _vec(e_str, xyz_str):
    return [float(e_str)] + [float(x) for x in xyz_str.split(",")]


def parse(n_max):
    a, knu, kmu, pN, ppi = [], [], [], [], []
    t0 = time.time()
    with open(LOG) as f:
        for line in f:
            if not line.startswith("RESGEN"):
                continue
            d = dict(_kv(t) for t in line.split()[1:])     # drop 'RESGEN' tag
            try:
                a.append(float(d["amps2"]))
                knu.append(_vec(d["liE"], d["li"]))
                kmu.append(_vec(d["loE"], d["lo"]))
                pN.append(_vec(d["hNE"], d["hN"]))
                ppi.append(_vec(d["hPE"], d["hP"]))
            except (KeyError, ValueError):
                continue
            if len(a) >= n_max:
                break
    print(f"  parsed {len(a)} events in {time.time()-t0:.0f}s", flush=True)
    return (np.array(a), np.array(knu), np.array(kmu), np.array(pN), np.array(ppi))


def my_amps2(knu, kmu, pN, ppi, nchunks):
    n = len(knu); chunk = (n + nchunks - 1) // nchunks
    # struck proton AT REST at lab origin (mono free-proton target, 939.565? no: proton mass)
    import adonis.xsec.constants as _C
    import os as _os
    _pm = _C.mN if _os.environ.get('PSTRUCK_MQE','0')=='1' else __import__('adonis.xsec.backend',fromlist=['MASS_PDG_PROTON']).MASS_PDG_PROTON
    pst = np.tile([_pm, 0.0, 0.0, 0.0], (n, 1))
    out = np.zeros(n); t0 = time.time()
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        out[sl] = np.asarray(exclusive_amps2_batch(
            knu[sl], kmu[sl], pst[sl], pN[sl], ppi[sl], +1, 211))  # itiz=+1 proton, pi+
        print(f"  [amps2] {min(i+chunk,n)}/{n}  {time.time()-t0:.0f}s", flush=True)
    return np.where(np.isfinite(out), out, 0.0)


def kine(knu, kmu, pN, ppi):
    q = knu - kmu
    Q2 = (q[:, 1:] ** 2).sum(1) - q[:, 0] ** 2
    pcm = pN + ppi
    W = np.sqrt(np.clip(pcm[:, 0] ** 2 - (pcm[:, 1:] ** 2).sum(1), 0, None))
    return Q2 * 1e-6, W            # Q2 in GeV^2, W in MeV


def report(name, x, r, ok, bins):
    print(f"\n{name}  (ratio my/ach, weighted-mean per bin):")
    c = 0.5 * (bins[1:] + bins[:-1])
    for i in range(len(c)):
        m = ok & (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() > 5:
            print(f"  {c[i]:9.3f}  n={m.sum():6d}  <my/ach>={np.mean(r[m]):.3f}  "
                  f"sum_my/sum_ach={r[m].dot(np.ones(m.sum()))/m.sum():.3f}  "
                  f"ratio_of_sums={x[m].size and np.sum(_ACH[m]*r[m])/np.sum(_ACH[m]):.3f}")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    # extra args: knob=value (e.g. pion_pole=2.0 idxp_start=1) -> assembly.DBG
    from adonis.primary.dcc import assembly as _asm
    for tok in sys.argv[3:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            _asm.DBG[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    print(f"=== resgen_ratio N={N}  DBG={_asm.DBG} ===", flush=True)
    global _ACH
    _ACH, knu, kmu, pN, ppi = parse(N)
    Q2, W = kine(knu, kmu, pN, ppi)
    print("computing my_amps2 ...", flush=True)
    my = my_amps2(knu, kmu, pN, ppi, nchunks)
    ok = (_ACH > 0) & (my > 0)
    r = np.where(ok, my / np.where(ok, _ACH, 1), 0.0)
    print(f"\noverall: n_ok={ok.sum()}  global sum_my/sum_ach={my[ok].sum()/_ACH[ok].sum():.4f}")
    report("Q2 [GeV^2]", Q2, r, ok, np.linspace(0, 1.2, 13))
    report("W [MeV]", W, r, ok, np.linspace(1080, 1500, 15))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
