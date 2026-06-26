"""Transport-SEQUENCING test: per-scatter physics is verified faithful (formation zone, k_F formula/
density/position, hard-cut Pauli on both outgoing, sigma, Gaussian prob, target selection all match
ACHILLES).  The residual QE multi-knockout-neutron deficit (~2.7%, bimodal soft-Pauli + hard-tail) must
then live in the ORDER ADoNIS processes scatters (pool global-lockstep + shared `consumed` + refill
queue) vs ACHILLES's recursive single-particle queue.  This isolates that by running BOTH transports on
the SAME ACHILLES input nucleus and comparing the cascade INTENSITY, not just the final state:

  - successful NN scatters / event           (ADoNIS chan in {1,2} nucleon segs  vs  ACHILLES not-blocked)
  - successful scatters that EJECT a neutron  (recoil/inelastic neutron)          per event
  - ADoNIS per-GENERATION scatter counts      (gen 0 primary vs gen>=1 sub-cascade) -- the sequencing
    signature: if gen-0 matches but gen>=1 is deficient, the pool suppresses the sub-cascade.

Run: NMAX=200000 LOGCAP=96 python -u scripts/cascade_seq_test.py _oracle_out/QE_cascdump_big.cdump
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus
from adonis.data.oracle.parse_cascadedump import parse_cascadedump, to_arrays

DUMP = sys.argv[1] if len(sys.argv) > 1 else "_oracle_out/QE_cascdump.cdump"
NMAX = int(os.environ.get("NMAX", "200000"))
LOGCAP = int(os.environ.get("LOGCAP", "96"))
TIMESTEP = os.environ.get("TIMESTEP", "0") == "1"   # 1 = ACHILLES-like time-sync (beta*step) stepping
STEP = float(os.environ.get("STEP", "0.04"))        # spatial step [fm] -- vary to probe step-granularity


def build_cfg(target="C"):
    tg = resolve_targets(target)[0][0]
    # physics reach = path_budget_R * radius (scheme-independent); max_steps is the 100k runaway tripwire.
    return DiscreteCascadeConfig(step=STEP, path_budget_R=3.0, seed=1,
                                 nn_inelastic=True, pauli=True, time_step=TIMESTEP,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def su_from_achilles(arr, key):
    n = arr["npos"].shape[0]
    return dict(npos=jnp.asarray(arr["npos"]), nmom=jnp.asarray(arr["nmom"]),
                nisp=jnp.asarray(arr["nisp"]), pos0=jnp.asarray(arr["prim_pos"]),
                consumed0=jnp.asarray(~arr["nmask"]), ch0=jnp.ones(n, jnp.int32), kp=key)


def main():
    print(f"parsing {DUMP} (NMAX={NMAX}) ...", flush=True)
    events = parse_cascadedump(DUMP)
    events = [e for e in events if e["proc"] == 0][:NMAX]          # QE only
    arr = to_arrays(events)
    n = arr["npos"].shape[0]
    w = arr["w"]; W = w.sum()
    print(f"  {n} QE events   stepping={'TIME-sync (beta*step)' if TIMESTEP else 'DISTANCE-sync (step)'}"
          f"   step={STEP} fm", flush=True)

    # ---- ACHILLES cascade intensity (from CDSCAT trace) ----
    ach_succ = np.zeros(n); ach_att = np.zeros(n); ach_blk = np.zeros(n)
    ach_succ_nrec = np.zeros(n)                                    # not-blocked scatters ejecting a recoil n
    for i, e in enumerate(events):
        for s in e["scat"]:
            ach_att[i] += 1
            if s["blocked"]:
                ach_blk[i] += 1
                continue
            ach_succ[i] += 1
            # recoil = struck (pid2) species for elastic; count struck-neutron not-blocked scatters
            if s["pid2"] == 2112:
                ach_succ_nrec[i] += 1

    # ---- ADoNIS cascade intensity (segment logger on IDENTICAL input) ----
    cfg = build_cfg("C")
    key = jax.random.PRNGKey(12345)
    su = su_from_achilles(arr, key)
    p_pi = jnp.zeros((n, 4)); p_N = jnp.asarray(arr["prim_p4"])
    ppid = jnp.zeros(n, jnp.int32); ipid = jnp.full(n, 2112, jnp.int32); Npid = jnp.full(n, 2212, jnp.int32)
    print(f"running ADoNIS cascade w/ segment logger (LOGCAP={LOGCAP}) ...", flush=True)
    log, counts, ovf = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, P=2, channel="qe",
                                       su_external=su, log_cap=LOGCAP)
    counts = np.asarray(counts)
    chan = np.asarray(log["chan"]); inc_pid = np.asarray(log["inc_pid"]); gen = np.asarray(log["gen"])
    n1_pid = np.asarray(log["n1_pid"]); n1_al = np.asarray(log["n1_al"])
    n2_pid = np.asarray(log["n2_pid"]); n2_al = np.asarray(log["n2_al"])
    rowmask = np.arange(chan.shape[1])[None, :] < counts[:, None]  # valid rows per event
    is_nuc = np.isin(inc_pid, (2212, 2112)) & rowmask
    succ = is_nuc & np.isin(chan, (1, 2))                          # successful nucleon scatter
    ado_succ = succ.sum(1).astype(float)
    ado_att = ado_succ.copy()                                      # logger does not record blocked attempts
    eject_n = succ & (((n1_pid == 2112) & (n1_al > 0)) | ((n2_pid == 2112) & (n2_al > 0)))
    ado_succ_nrec = eject_n.sum(1).astype(float)
    ado_g0 = (succ & (gen == 0)).sum(1).astype(float)
    ado_g1 = (succ & (gen >= 1)).sum(1).astype(float)
    print(f"  log overflow events: {int(np.asarray(ovf).sum()) if np.ndim(ovf) else int(ovf)}  "
          f"max rows/event: {counts.max()} (cap {LOGCAP})", flush=True)

    def m(x): return float((x * w).sum() / W)
    print("\n=== CASCADE INTENSITY on IDENTICAL input (weighted means/event) ===", flush=True)
    print(f"  successful NN scatters:        ADoNIS {m(ado_succ):.4f}   ACHILLES {m(ach_succ):.4f}   "
          f"ACH/ADO {m(ach_succ)/max(m(ado_succ),1e-9):.3f}", flush=True)
    print(f"  scatters ejecting a neutron:   ADoNIS {m(ado_succ_nrec):.4f}   ACHILLES {m(ach_succ_nrec):.4f}   "
          f"ACH/ADO {m(ach_succ_nrec)/max(m(ado_succ_nrec),1e-9):.3f}", flush=True)
    print(f"  ACHILLES attempts (incl blkd): {m(ach_att):.4f}   blocked frac {m(ach_blk)/max(m(ach_att),1e-9):.3f}", flush=True)
    print(f"\n  ADoNIS successful scatters by GENERATION:", flush=True)
    print(f"    gen 0 (primary p):  {m(ado_g0):.4f}", flush=True)
    print(f"    gen>=1 (sub-casc):  {m(ado_g1):.4f}", flush=True)
    print(f"    ADoNIS gen0 vs ACHILLES total-1st? (sanity)", flush=True)

    # distribution of successful scatters/event
    print("\n  successful-NN-scatters/event distribution (weighted fraction):", flush=True)
    print("   k     ADoNIS      ACHILLES    ACH/ADO", flush=True)
    for k in range(6):
        fa = float((w * (ado_succ == k)).sum() / W); fh = float((w * (ach_succ == k)).sum() / W)
        print(f"   {k}   {fa:.5f}    {fh:.5f}    {fh/max(fa,1e-9):.3f}", flush=True)
    print(f"   >=6 ADoNIS {float((w*(ado_succ>=6)).sum()/W):.5f}   ACHILLES {float((w*(ach_succ>=6)).sum()/W):.5f}", flush=True)


if __name__ == "__main__":
    main()
