"""Decisive: strip the amplitude from ACHILLES's physical events to expose its effective dPhi3.
ACHILLES hepmc events ~ ach_amps2 * dPhi3.  Reweight by 1/my_amps2 (= 1/ach_amps2 on the dump-
validated function) -> ~ dPhi3.  Compare that to a flat-Dalitz ground-truth dPhi3 and to my
isotropic forward's dPhi3.  If ACHILLES's stripped dPhi3 != flat-Dalitz -> ACHILLES phase space
is non-standard (the bug).  If it matches -> amplitude differs off-dump.  Chunked + progress."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import scripts.test_3body_q2_measure as T
import scripts.mono_dist_compare as M
from adonis.xsec.dcc_current import exclusive_amps2_batch


def my_amps2_on(knu, kmu, pN, ppi, nchunks=10):
    n = len(knu); chunk = (n + nchunks - 1) // nchunks
    out = np.zeros(n); pst = np.tile(T.p_st, (n, 1)); t0 = time.time()
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        out[sl] = np.asarray(exclusive_amps2_batch(knu[sl], kmu[sl], pst[sl], pN[sl], ppi[sl], +1, 211))
        print(f"  [amps2] {min(i+chunk,n)}/{n}  {time.time()-t0:.0f}s", flush=True)
    return np.where(np.isfinite(out), out, 0.0)


def main():
    print("parsing ACHILLES mono hepmc ...", flush=True)
    aknu, akmu, apN, appi = M.parse_ach()
    aQ2, aW, _, _ = M.variables(aknu, akmu, apN, appi)
    print(f"  {len(aknu)} events; computing my_amps2 on them ...", flush=True)
    a_amp = my_amps2_on(aknu, akmu, apN, appi)
    ok = a_amp > 0
    wstrip = np.where(ok, 1.0 / np.where(ok, a_amp, 1.0), 0.0)      # 1/amps2 -> dPhi3 shape

    # flat-Dalitz ground-truth dPhi3 (pure phase space, equal weight)
    rng = np.random.default_rng(0)
    qf = T.flat_dalitz(3_000_000, rng)                              # Q2 only (pure PS)
    # need W for flat-Dalitz too -> recompute with full kinematics
    kmu, pN, ppi, okf = T.flat_dalitz_amps2(2_000_000, rng)
    knu_f = np.tile(T.k_nu, (len(kmu), 1))
    fQ2, fW, _, _ = M.variables(knu_f[okf], kmu[okf], pN[okf], ppi[okf])

    def cmp(name, av, aw_, fv, bins):
        ha, _ = np.histogram(av, bins=bins, weights=aw_); ha = ha / ha.sum()
        hf, _ = np.histogram(fv, bins=bins); hf = hf / hf.sum()
        c = 0.5 * (bins[1:] + bins[:-1])
        print(f"\n{name}: ACHILLES-stripped-dPhi3 vs flat-Dalitz-dPhi3 (ratio ACH/flat):")
        for i in range(len(c)):
            if hf[i] > 0:
                print(f"  {c[i]:9.3f}   ACHstr {ha[i]:.4f}  flat {hf[i]:.4f}   ratio {ha[i]/hf[i]:.3f}")
    cmp("Q2 [GeV^2]", aQ2[ok], wstrip[ok], fQ2, np.linspace(0, 1.0, 11))
    cmp("W [MeV]", aW[ok], wstrip[ok], fW, np.linspace(1080, 1560, 11))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
