"""STANDALONE Cat-3 (fixed-cap) stress test -- does any engine buffer truncate on the physics sample?

Runs the REAL pool cascade (cascade_nucleus) on the HIGHEST-A material we use (Ar=40) with the actual
T2K flux, and reports for each counter-equipped cap both the overflow counter AND the per-event activity
DISTRIBUTION vs the cap (so the MARGIN is visible even at 0 overflow):
  * pool stack width M=P (+ out-buffer M_out) -> ofl = sofl+oofl ; swept over P
  * log_cap L -> segments/event distribution (counts) vs L
  * record caps Kp,Kn (with_rec) -> pion-hits/event, nucleon-steps/event vs the passed Kp,Kn
NOTE: _KSLAB (nearest in-slab nucleons) has NO counter (top_k silently drops the 4th+); it needs a
1-line candidate-count export -> done separately in a git worktree, not here.

Run: python -u scripts/test_cascade_caps.py [C|Ar] [N] [chunk]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT = sys.argv[1] if len(sys.argv) > 1 else "Ar"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec, qe_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density, _KSLAB
from adonis.fsi.cascade_full import cascade_nucleus
LCAP = 128                                                        # generous, to MEASURE the true counts tail


def cfg_for(tg):
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    return DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs), radius


def dist(name, arr, cap):
    a = np.asarray(arr)
    pct = np.percentile(a, [50, 99, 99.9, 100])
    print(f"  {name:28s} cap={cap:<4} | median {pct[0]:.0f}  p99 {pct[1]:.0f}  p99.9 {pct[2]:.0f}  "
          f"MAX {pct[3]:.0f}  -> margin {cap/max(pct[3],1):.2f}x  overflow_events={int((a>cap).sum())}")


def main():
    tg = resolve_targets(MAT)[0][0]
    cfg, radius = cfg_for(tg)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    print(f"[Cat-3 caps] {MAT} A={tg.A} radius={radius:.2f} max_steps={cfg.max_steps} N={N}")
    print(f"  production caps: pool P (swept), log_cap=64, _KSLAB={_KSLAB}")

    all_counts = []                                              # segments/event (vs log_cap)
    ofl_by_P = {8: 0, 12: 0}                                      # production P=12 + one lower for margin
    for sd in range((N + CHUNK - 1) // CHUNK):
        nthis = min(CHUNK, N - sd * CHUNK)
        e = res_xsec.generate(nthis, seed=sd, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
        rw = np.asarray(e["w"]); rs = rw > 0
        ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
        pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
        ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
        k = jax.random.PRNGKey(31 + sd)
        # (1) segments/event via the logged path (generous L to see the true tail)
        _lg, counts, logofl = cascade_nucleus(ppi, pN, ppid, ipid, Npid, cfg, k, P=12, channel="res", log_cap=LCAP)
        all_counts.append(np.asarray(counts))
        # (2) pool-M overflow at production P=12 + one lower P for the margin (no log)
        for P in ofl_by_P:
            *_, ofl, _c = cascade_nucleus(ppi, pN, ppid, ipid, Npid, cfg, k, P=P, channel="res")
            ofl_by_P[P] += int(ofl)
        print(f"  chunk {sd+1}: +{int(rs.sum())} events (logofl={int(logofl)})", flush=True)

    counts = np.concatenate(all_counts)
    print(f"\n=== {MAT} cap margins (RES primaries, N={len(counts)}) ===")
    dist("segments/event", counts, 64)                          # vs production log_cap=64
    print("  pool stack overflow (sofl+oofl) by P:")
    for P, o in sorted(ofl_by_P.items()):
        print(f"    P={P:<3} -> overflow={o}  {'<-- production' if P==12 else ''}")
    print("  (record caps Kp/Kn deferred -- tuning-path only; _KSLAB via scripts/test_kslab_ab.py)")


if __name__ == "__main__":
    main()
